"""
scraper_final.py
================
BDS23124 – Johor Property DSS
PropertyGuru 爬虫

P4 修复：
  ✅ extract_location() 无 fallback，找不到返回 None
  ✅ 加入 Bedrooms / Bathrooms 真实抓取
  ✅ Size 字符串清洗成 Size_SQFT 数字
  ✅ 无硬编码路径（用相对路径）
  ✅ 无 version_main 硬编码
  ✅ Price 上限 5,000,000
  ✅ 无法识别 Location 的条目直接跳过

执行方法：
    python scraper_final.py
输出：
    johor_scraped_clean.csv
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time
import re
import os

# ============================================================
# 配置（路径全部用相对路径或变量，不硬编码）
# ============================================================
OUTPUT_CSV   = 'johor_scraped_clean.csv'
TOTAL_PAGES  = 20
WAIT_PAGE    = 12   # 页面加载等待秒数
WAIT_BETWEEN = 3    # 翻页间隔秒数

# ============================================================
# 关键词过滤
# ============================================================
TITLE_KEYWORDS = [
    'terrace', 'house', 'home', 'semi-d', 'bungalow',
    'condo', 'condominium', 'apartment', 'flat', 'villa',
    'studio', 'residence', 'residences', 'suite', 'suites',
    'storey', 'story', 'bedroom', 'renovated', 'corner',
    'link', 'cluster', 'townhouse', 'height', 'park'
]

SKIP_KEYWORDS = [
    'whatsapp', 'call', 'contact', 'agent', 'listed on',
    'starting from', 'psf', 'enquire', 'official listing'
]

# ============================================================
# 合法 Location 列表（匹配不到就 None，不用 fallback）
# ============================================================
LOCATIONS = [
    'Johor Bahru', 'Skudai', 'Iskandar Puteri', 'Nusajaya',
    'Kulai', 'Kluang', 'Batu Pahat', 'Johor Jaya',
    'Mount Austin', 'Bukit Indah', 'Taman Molek',
    'Gelang Patah', 'Pontian', 'Segamat', 'Muar',
    'Permas Jaya', 'Pasir Gudang', 'Ulu Tiram',
    # 常见子地区 → 映射到合法 Location
    'Tebrau', 'Tampoi', 'Perling', 'Masai',
    'Senai', 'Plentong', 'Pengerang',
]

# 子地区 → 合法 Location 映射
SUBLOCATION_MAP = {
    'nusajaya':     'Iskandar Puteri',
    'tebrau':       'Johor Bahru',
    'tampoi':       'Johor Bahru',
    'perling':      'Johor Bahru',
    'masai':        'Pasir Gudang',
    'senai':        'Kulai',
    'plentong':     'Johor Bahru',
    'pengerang':    'Pasir Gudang',
    'setia indah':  'Johor Bahru',
    'setia tropika':'Johor Bahru',
    'horizon hills':'Iskandar Puteri',
    'kota tinggi':  'Kluang',
}

# ============================================================
# Helper functions
# ============================================================
def is_real_title(text):
    text_lower = text.lower()
    if len(text) < 10 or len(text) > 150:
        return False
    if text.strip().isdigit():
        return False
    if ',' in text:
        return False
    if any(k in text_lower for k in SKIP_KEYWORDS):
        return False
    return any(k in text_lower for k in TITLE_KEYWORDS)


def clean_price(price_str):
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return int(cleaned) if cleaned else 0


def get_prop_type(title):
    t = title.lower()
    if 'bungalow' in t:
        return 'Bungalow'
    elif 'semi-d' in t or 'semi d' in t:
        return 'Semi-D'
    elif 'condo' in t or 'condominium' in t:
        return 'Condominium'
    elif 'apartment' in t or 'flat' in t:
        return 'Apartment'
    elif 'terrace' in t or 'storey' in t or 'story' in t:
        return 'Terrace House'
    elif 'studio' in t:
        return 'Apartment'
    elif 'villa' in t:
        return 'Bungalow'
    elif 'residence' in t or 'suite' in t:
        return 'Serviced Residence'
    else:
        return 'Residential'


def extract_location(address):
    """
    严格匹配，找不到返回 None。
    不使用任何 fallback。
    """
    addr_lower = address.lower()

    # 先检查子地区映射
    for subloc, target in SUBLOCATION_MAP.items():
        if subloc in addr_lower:
            return target

    # 再检查标准 Location 列表
    for loc in LOCATIONS:
        if loc.lower() in addr_lower:
            # 过滤掉 Nusajaya（映射到 Iskandar Puteri）
            if loc == 'Nusajaya':
                return 'Iskandar Puteri'
            return loc

    return None  # ✅ 找不到就 None，不用 fallback


def parse_number_before_label(lines, label):
    """
    在 lines 里找 label（如 'bed', 'bath'），
    取前一行的纯数字作为该值。
    """
    for i, line in enumerate(lines):
        if line.strip().lower() == label and i > 0:
            prev = lines[i - 1].strip().replace(',', '')
            if prev.isdigit():
                val = int(prev)
                if 1 <= val <= 20:
                    return val
    return None


def parse_size_sqft(lines):
    """
    从 lines 里提取面积数字（sqft）。
    """
    for line in lines:
        if 'sqft' in line.lower():
            # 格式：'1,540 sqft' 或 '(1,540 sqft)'
            match = re.search(r'([\d,]+)\s*sqft', line, re.IGNORECASE)
            if match:
                val = int(match.group(1).replace(',', ''))
                if 200 <= val <= 20000:
                    return float(val)
    return None


# ============================================================
# 抓取单页
# ============================================================
def scrape_page(driver):
    results = []
    listings = driver.find_elements(
        By.XPATH, "//div[contains(@class, 'listing-card')]"
    )

    for item in listings:
        try:
            lines = [l.strip() for l in item.text.split('\n') if l.strip()]

            # 找价格
            price_line = None
            for line in lines:
                if line.startswith('RM') and 'psf' not in line.lower():
                    price_line = line
                    break
            if not price_line:
                continue

            price_cleaned = clean_price(price_line)
            if price_cleaned < 100_000 or price_cleaned > 5_000_000:  # ✅ 上限 5M
                continue

            # 找标题
            title = None
            for line in lines:
                if is_real_title(line):
                    title = line
                    break
            if not title:
                for line in lines:
                    if any(k in line.lower() for k in
                           ['house', 'condo', 'apartment',
                            'residence', 'bungalow', 'semi-d']):
                        if len(line) < 60 and ',' not in line:
                            title = line
                            break
            if not title:
                continue

            # 找地址
            address = ''
            for line in lines:
                if ',' in line and 'RM' not in line and 'sqft' not in line.lower():
                    if len(line) > 10:
                        address = line
                        break

            # ✅ 严格 Location 匹配，找不到就跳过
            location = extract_location(address)
            if location is None:
                continue

            # ✅ 真实 Bedrooms
            bedrooms = parse_number_before_label(lines, 'bed')
            if bedrooms is None:
                # 尝试从标题提取 (e.g. "3-bedroom")
                m = re.search(r'(\d+)[- ](?:bedroom|room|bed)', title, re.IGNORECASE)
                if m:
                    bedrooms = int(m.group(1))

            # ✅ 真实 Bathrooms
            bathrooms = parse_number_before_label(lines, 'bath')

            # ✅ 真实 Size_SQFT（数字）
            size_sqft = parse_size_sqft(lines)

            results.append({
                'Property_Title': title,
                'Price_Raw':      price_line,
                'Price_Cleaned':  price_cleaned,
                'Location':       location,
                'Address':        address,
                'Size_SQFT':      size_sqft if size_sqft else 0.0,
                'Bedrooms':       bedrooms,
                'Bathrooms':      bathrooms,
                'Property_Type':  get_prop_type(title),
                'Source':         'PropertyGuru',
                'Scrape_Date':    time.strftime('%Y-%m-%d'),
            })

        except Exception:
            continue

    return results


# ============================================================
# 主程序
# ============================================================
def main():
    # ✅ 无 version_main 硬编码
    options = uc.ChromeOptions()
    driver  = uc.Chrome(options=options)

    all_data = []

    try:
        for page in range(1, TOTAL_PAGES + 1):
            url = (f"https://www.propertyguru.com.my/property-for-sale/{page}"
                   f"?freetext=Johor")
            print(f"正在抓第 {page}/{TOTAL_PAGES} 页...", end=" ")

            driver.get(url)
            time.sleep(WAIT_PAGE)

            page_results = scrape_page(driver)
            all_data.extend(page_results)
            print(f"这页抓到 {len(page_results)} 条，累计 {len(all_data)} 条")

            time.sleep(WAIT_BETWEEN)

    finally:
        driver.quit()

    # ============================================================
    # 清理和保存
    # ============================================================
    if all_data:
        df = pd.DataFrame(all_data)

        # 去重
        df = df.drop_duplicates(subset=['Property_Title', 'Price_Cleaned'])
        df = df.reset_index(drop=True)

        # ✅ 用相对路径保存
        df.to_csv(OUTPUT_CSV, index=False)

        print(f"\n✅ 完成！共 {len(df)} 条干净数据")
        print(f"\n地区分布:")
        print(df['Location'].value_counts())
        print(f"\n类型分布:")
        print(df['Property_Type'].value_counts())
        print(f"\n有 Bedrooms 数据: {df['Bedrooms'].notna().sum()} 条")
        print(f"有 Bathrooms 数据: {df['Bathrooms'].notna().sum()} 条")
        print(f"有 Size_SQFT 数据: {(df['Size_SQFT'] > 0).sum()} 条")
    else:
        print("没有抓到数据")


if __name__ == '__main__':
    main()
