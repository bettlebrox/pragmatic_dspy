from firecrawl import FirecrawlApp
import json
from pprint import pprint
import os
import dotenv

dotenv.load_dotenv("local.env")


def crawl(url: str):
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    domain = url.split("//")[-1].split("/")[0].split(".")[0]
    # Crawl a website:
    crawl_status = app.crawl_url(
        url,
        params={"limit": 200, "scrapeOptions": {"formats": ["html"]}},
        poll_interval=5,
    )
    print(crawl_status)
    if not os.path.exists(f"data/{domain}"):
        os.makedirs(f"data/{domain}")
    # Save each crawled item to a separate JSON file
    for i, item in enumerate(crawl_status["data"]):
        pprint(item)
        filename = f"data/{domain}/crawl_item_{i}.json"
        with open(filename, "w") as f:
            json.dump(item, f, indent=4)
        print(f"Saved {filename}")
