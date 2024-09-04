# %%
import sys
import requests


def get_book_info(isbn):
    url = f"https://api.openbd.jp/v1/get?isbn={isbn}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            book_data = response.json()
            if book_data:
                return book_data
            else:
                print("書籍情報が見つかりませんでした。")
        else:
            print(f"エラーコード {response.status_code} が返されました。")
    except requests.exceptions.RequestException as e:
        print(f"リクエストエラー: {e}")


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        isbn = "9784000069731"  # ISBNを指定
    else:
        isbn = sys.argv[1]
    res = get_book_info(isbn)
    try:
        print(res[0]["summary"]["title"])
    except:
        print(res)
# %%
