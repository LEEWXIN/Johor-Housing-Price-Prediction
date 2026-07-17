"""
scraper_mudah.py
================
BDS23124 – Johor Property DSS
Mudah.my 爬虫

P4 修复：
  ✅ extract_location() 无 fallback，找不到返回 None
  ✅ Bedrooms/Bathrooms 找不到时填 None（不填 0）
  ✅ 无硬编码路径（用相对路径）
  ✅ 无 version_main 硬编码
  ✅ Price 上限 5,000,000
  ✅ 无法识别 Location 的条目直接跳过

执行方法：
    python scraper_mudah.py
输出：
    mudah_scraped.csv
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time
import re

# ============================================================
# 配置
# ============================================================
OUTPUT_CSV   = 'mudah_scraped.csv'
TOTAL_PAGES  = 50
WAIT_PAGE    = 12
WAIT_BETWEEN = 3

# ============================================================
# 合法 Location 列表 + 子地区映射
# ============================================================
LOCATIONS = [
    'Johor Bahru', 'Skudai', 'Iskandar Puteri',
    'Kulai', 'Kluang', 'Batu Pahat', 'Johor Jaya',
    'Mount Austin', 'Bukit Indah', 'Taman Molek',
    'Gelang Patah', 'Pontian', 'Segamat', 'Muar',
    'Permas Jaya', 'Pasir Gudang', 'Ulu Tiram',
]

SUBLOCATION_MAP = {
    'nusajaya':         'Iskandar Puteri',
    'tebrau':           'Johor Bahru',
    'tampoi':           'Johor Bahru',
    'perling':          'Johor Bahru',
    'masai':            'Pasir Gudang',
    'senai':            'Kulai',
    'plentong':         'Johor Bahru',
    'pengerang':        'Pasir Gudang',
    'setia indah':      'Johor Bahru',
    'setia tropika':    'Johor Bahru',
    'horizon hills':    'Iskandar Puteri',
    'kota tinggi':      'Kluang',
    'simpang renggam':  'Kulai',
    'ayer hitam':       'Batu Pahat',
    'pekan nanas':      'Pontian',
    'parit raja':       'Batu Pahat',
    'yong peng':        'Batu Pahat',
    'rengit':           'Batu Pahat',
    'tangkak':          'Muar',
    'pagoh':            'Muar',
    'bakri':            'Muar',
    'mersing':          'Kluang',
}


# ============================================================
# Helper functions
# ============================================================
def clean_price(price_str):
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return int(cleaned) if cleaned else 0


def get_prop_type(text):
    t = text.lower()
    if 'bungalow' in t or 'detached' in t:
        return 'Bungalow'
    elif 'semi-d' in t or 'semi d' in t:
        return 'Semi-D'
    elif 'condo' in t or 'condominium' in t:
        return 'Condominium'
    elif 'apartment' in t or 'flat' in t:
        return 'Apartment'
    elif 'terrace' in t or 'storey' in t or 'story' in t or 'link' in t:
        return 'Terrace House'
    elif 'studio' in t:
        return 'Apartment'
    elif 'villa' in t:
        return 'Bungalow'
    elif 'residence' in t or 'suite' in t:
        return 'Serviced Residence'
    else:
        return 'Residential'


def extract_location(text):
    """
    严格匹配，找不到返回 None。
    不使用任何 fallback。
    """
    text_lower = text.lower()

    # 先检查子地区映射（长词优先）
    for subloc, target in SUBLOCATION_MAP.items():
        if subloc in text_lower:
            return target

    # 再检查标准 Location
    for loc in LOCATIONS:
        if loc.lower() in text_lower:
            return loc

    return None  # ✅ 找不到就 None，不用 fallback


def parse_number_before_label(lines, label):
    """
    在 lines 里找 label（'bed' 或 'bath'），
    取前一行的纯数字。找不到返回 None（不返回 0）。
    """
    for i, line in enumerate(lines):
        if line.strip().lower() == label and i > 0:
            prev = lines[i - 1].strip().replace(',', '')
            if prev.isdigit():
                val = int(prev)
                if 1 <= val <= 20:
                    return val
    return None  # ✅ 找不到返回 None，不填 0


def parse_size_sqft(lines):
    """
    从 lines 里找面积数字（sq.ft）。
    """
    for i, line in enumerate(lines):
        if 'sq.ft' in line.lower() and i > 0:
            prev = lines[i - 1].replace(',', '').strip()
            if prev.isdigit():
                val = int(prev)
                if 200 <= val <= 20000:
                    return float(val)
    return None


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
            url = f"https://www.mudah.my/johor/houses-for-sale?o={page}"
            print(f"正在抓第 {page}/{TOTAL_PAGES} 页...", end=" ")

            driver.get(url)
            time.sleep(WAIT_PAGE)

            price_elements = driver.find_elements(
                By.XPATH, "//span[contains(@class, 'currPrice')]"
            )

            page_count = 0
            for price_el in price_elements:
                try:
                    price_text = price_el.text.strip()
                    if not price_text.startswith('RM'):
                        continue

                    price_cleaned = clean_price(price_text)
                    if price_cleaned < 100_000 or price_cleaned > 5_000_000:  # ✅ 上限 5M
                        continue

                    # 往上找父元素（包含完整卡片）
                    card      = price_el.find_element(By.XPATH, "../..")
                    card_text = card.text.strip()
                    if not card_text:
                        continue

                    lines = [l.strip() for l in card_text.split('\n') if l.strip()]

                    # 房产类型（第一行）
                    prop_type_raw = lines[0] if lines else ""
                    prop_type     = get_prop_type(prop_type_raw)

                    # 找地址（含 johor 的行）
                    address = ""
                    for line in lines:
                        if 'johor' in line.lower() and ',' in line:
                            address = line
                            break

                    # ✅ 严格 Location 匹配，找不到就跳过
                    location = extract_location(card_text + ' ' + address)
                    if location is None:
                        continue

                    # ✅ 真实 Size_SQFT（None 如果找不到）
                    size_sqft = parse_size_sqft(lines)

                    # ✅ 真实 Bedrooms（None 如果找不到，不填 0）
                    bedrooms  = parse_number_before_label(lines, 'bed')

                    # ✅ 真实 Bathrooms（None 如果找不到，不填 0）
                    bathrooms = parse_number_before_label(lines, 'bath')

                    # 标题（最后一行中最长的描述）
                    title = ""
                    for line in reversed(lines):
                        if (len(line) > 15 and 'RM' not in line
                                and 'johor' not in line.lower()
                                and 'bed' not in line.lower()
                                and 'bath' not in line.lower()):
                            title = line
                            break
                    if not title:
                        title = prop_type_raw

                    all_data.append({
                        'Property_Title': title,
                        'Price_Raw':      price_text,
                        'Price_Cleaned':  price_cleaned,
                        'Location':       location,
                        'Address':        address,
                        'Size_SQFT':      size_sqft,    # None if not found
                        'Bedrooms':       bedrooms,     # None if not found
                        'Bathrooms':      bathrooms,    # None if not found
                        'Property_Type':  prop_type,
                        'Source':         'Mudah',
                        'Scrape_Date':    time.strftime('%Y-%m-%d'),
                    })
                    page_count += 1

                except Exception:
                    continue

            print(f"这页抓到 {page_count} 条，累计 {len(all_data)} 条")
            time.sleep(WAIT_BETWEEN)

    finally:
        driver.quit()

    # ============================================================
    # 保存
    # ============================================================
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=['Property_Title', 'Price_Cleaned'])
        df = df.reset_index(drop=True)

        # ✅ 用相对路径保存
        df.to_csv(OUTPUT_CSV, index=False)

        print(f"\n✅ Mudah 完成！共 {len(df)} 条")
        print(f"\n有 Bedrooms 数据: {df['Bedrooms'].notna().sum()} 条")
        print(f"有 Bathrooms 数据: {df['Bathrooms'].notna().sum()} 条")
        print(f"有 Size_SQFT 数据: {df['Size_SQFT'].notna().sum()} 条")
        print(f"\n地区分布:")
        print(df['Location'].value_counts())
        print(f"\n类型分布:")
        print(df['Property_Type'].value_counts())
    else:
        print("没有抓到数据")


if __name__ == '__main__':
    main()
