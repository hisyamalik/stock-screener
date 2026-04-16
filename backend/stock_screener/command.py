from screener_id import IndonesianStockScreener, ScreenerConfig


def run_daily_screener(max_symbols=None, top_n: int = 30, send_telegram: bool = True):
    """Run the latest IDX daily screener workflow end-to-end."""
    config = ScreenerConfig()
    screener = IndonesianStockScreener(config)

    print("Refreshing IDX symbol universe...")
    screener.load_all_idx_symbols()

    print("Running daily technical + volume screen...")
    results = screener.screen_stocks(max_symbols=max_symbols)

    if not results:
        print("No candidates returned from screen.")
        return []

    screener.display_report(top_n=top_n)
    screener.export_report(csv_path="screener_report.csv", json_path="screener_report.json")

    if send_telegram:
        screener.send_telegram_report(csv_path="screener_report.csv", json_path="screener_report.json")

    return results


def main():
    run_daily_screener(max_symbols=None, top_n=30, send_telegram=True)


if __name__ == "__main__":
    main()
