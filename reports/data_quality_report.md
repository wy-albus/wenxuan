# Data Quality Report

## Overall

- Zip files: 7
- CSV files: 7
- Total raw rows: 21771594
- Time range: 2023-01-01 to 2026-06-21
- qty < 0 rows: 205810

## File Summary

| zip                   | csv                                | encoding   |    rows | columns                                                                                                                                                                                                                                   | missing_fields   | period_range            | isbn_missing_rate   | gds_no_missing_rate   |
|:----------------------|:-----------------------------------|:-----------|--------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|:------------------------|:--------------------|:----------------------|
| 20230101-20230630.zip | dmr_ls_rd_sal_dtl_202606231816.csv | gbk        | 3584789 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2023-01-01 - 2023-06-30 | 0.00%               | 0.00%                 |
| 20230701-20231231.zip | dmr_ls_rd_sal_dtl_202606231821.csv | gbk        | 3760665 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2023-07-01 - 2023-12-31 | 0.00%               | 0.00%                 |
| 20240101-20240630.zip | dmr_ls_rd_sal_dtl_202606231825.csv | gbk        | 3118204 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2024-01-01 - 2024-06-30 | 0.00%               | 0.00%                 |
| 20240701-20241231.zip | dmr_ls_rd_sal_dtl_202606231829.csv | gbk        | 3253070 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2024-07-01 - 2024-12-31 | 0.00%               | 0.00%                 |
| 20250101-20250630.zip | dmr_ls_rd_sal_dtl_202606231834.csv | gbk        | 2740816 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2025-01-01 - 2025-06-30 | 0.00%               | 0.00%                 |
| 20250701-20251231.zip | dmr_ls_rd_sal_dtl_202606231837.csv | gbk        | 3047334 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2025-07-01 - 2025-12-31 | 0.00%               | 0.00%                 |
| 20260101-20260622.zip | dmr_ls_rd_sal_dtl_202606231842.csv | gbk        | 2266716 | period, sal_chnl, bus_type, sal_type, lbus_type, oln_or_ofln, site_no, blt_site_no, cust_no, 2_lvel_cust_code, gds_no, isbn, rtn_flag, rtn_rfrc_vchr, vend_no, gds_ctgry_4_lvel, gds_ctgry_3_lvel, gds_ctgry_5_lvel, price, qty, tlp, tsp | None             | 2026-01-01 - 2026-06-21 | 0.00%               | 0.00%                 |

## Core Field Missing Rates

| field | missing_count | missing_rate |
|---|---:|---:|
| period | 0 | 0.00% |
| sal_chnl | 0 | 0.00% |
| bus_type | 0 | 0.00% |
| sal_type | 0 | 0.00% |
| lbus_type | 0 | 0.00% |
| oln_or_ofln | 0 | 0.00% |
| site_no | 0 | 0.00% |
| blt_site_no | 19403504 | 89.12% |
| gds_no | 0 | 0.00% |
| isbn | 0 | 0.00% |
| rtn_flag | 21565784 | 99.05% |
| vend_no | 3 | 0.00% |
| gds_ctgry_3_lvel | 0 | 0.00% |
| gds_ctgry_4_lvel | 0 | 0.00% |
| gds_ctgry_5_lvel | 0 | 0.00% |
| price | 0 | 0.00% |
| qty | 0 | 0.00% |
| tlp | 0 | 0.00% |
| tsp | 0 | 0.00% |

## Conversion Failures

| field | failed_or_missing_count | rate |
|---|---:|---:|
| period | 0 | 0.00% |
| qty | 0 | 0.00% |
| price | 0 | 0.00% |
| tlp | 0 | 0.00% |
| tsp | 0 | 0.00% |

## oln_or_ofln Values

| value | count |
|---|---:|
| ofln | 19260508 |
| oln | 2511086 |

## rtn_flag Values

| value | count |
|---|---:|
| (blank) | 21565784 |
| X | 205810 |
