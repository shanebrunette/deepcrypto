#!/usr/bin/python
# original source:
# https://github.com/jhogan4288/coinmarketcap-history
# /blob/master/coinmarketcap_usd_history.py

import csv
import sys
import re
from urllib.request import urlopen


def get_options(coin):
  # yyyymmdd
  start_date = '20140101'
  end_date   = '20180331'
  coin   = coin.lower()
  return coin, start_date, end_date


def download_data(coin, start_date, end_date):
  url = 'https://coinmarketcap.com/currencies/' 
        + coin + '/historical-data/' + '?start=' \
        + start_date + '&end=' + end_date
  try:
    page = urlopen(url,timeout=10)
    if page.getcode() != 200:
      raise Exception('Failed to load page') 
    html = page.read().decode('utf-8')
    page.close()
  except Exception as e:
    print('Error fetching price data from ' + url)
    
    if hasattr(e, 'message'):
      print("Error message: " + e.message)
    else:
      print(e)
      sys.exit(1)

  return html


def extract_data(html):
  head = re.search(r'<thead>(.*)</thead>', html, re.DOTALL).group(1)
  header = re.findall(r'<th .*>([\w ]+)</th>', head)

  body = re.search(r'<tbody>(.*)</tbody>', html, re.DOTALL).group(1)
  raw_rows = re.findall(
          r'<tr[^>]*>' + r'\s*<td[^>]*>([^<]+)</td>'*7 + r'\s*</tr>', body)

  # strip commas
  rows = []
  for row in raw_rows:
    row = [ field.translate(str.maketrans('','', ',')) for field in row ]
    rows.append(row)

  return header, rows


def write_csv_data(header, rows, coin):
  with open(f'../data/raw/{coin}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for row in rows:
      writer.writerow(row)


def scrape(coin):
  print(coin)
  coin, start_date, end_date = get_options(coin)
  html = download_data(coin, start_date, end_date)
  header, rows = extract_data(html) 
  write_csv_data(header, rows, coin)


def main():
  if sys.argv[1]:
    coin = sys.argv[1]
    scrape(coin)
  else:
    with open('../data/all_coins.txt', 'r') as fp:
      for line in fp:
        coin = line.strip()
        scrape(coin)


if __name__ == '__main__':
  main()

