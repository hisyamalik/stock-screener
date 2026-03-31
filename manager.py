from screener_id import IndonesianStockScreener


def refresh_idx_universe(print_preview: int = 20):
    """Refresh and return latest IDX symbol universe from all configured online sources."""
    screener = IndonesianStockScreener()
    symbols = screener.load_all_idx_symbols()

    print(f"Universe loaded: {len(symbols)} symbols")
    if symbols:
        preview = symbols[:print_preview]
        print(f"Preview ({len(preview)}): {preview}")

    return symbols


def export_universe_csv(path: str = "idx_universe.csv"):
    """Refresh universe and export to CSV for audit/reference."""
    symbols = refresh_idx_universe(print_preview=10)
    with open(path, "w", encoding="utf-8") as f:
        f.write("symbol\n")
        for symbol in symbols:
            f.write(f"{symbol}\n")
    print(f"Universe exported to {path}")


if __name__ == "__main__":
    export_universe_csv()
