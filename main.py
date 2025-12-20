def main():
    # Backwards-compatible entrypoint for `python main.py`.
    from insights.cli import main as insights_main

    insights_main()


if __name__ == "__main__":
    main()
