"""Display the saved top-five recommendations for three MovieLens users."""
import json
from pathlib import Path

profiles = json.loads((Path(__file__).resolve().parents[1] / 'public/recommendations.json').read_text())
for user_id in (1, 42, 100):
    print(f'\nUser {user_id}')
    for index, item in enumerate(profiles[str(user_id)]['recommendations'], 1):
        print(f"{index}. {item['title']} | estimated rating {item['score']:.2f}/5")
