import requests
import json 

with open('config.json') as json_file:
    config = json.load(json_file)
print(config)
COOKIES = {
    "Cookie": config["mainCookie"]
},
HEADERS = {
    'User-Agent': config["useragent"],
    'Referer': 'https://csgoempire.com/withdraw',
    'Accept': '/',
    'Connection': 'keep-alive',
}
cok = {'PHPSESSID':'uaj9ohe1006qdd85oinck0i3o6','do_not_share_this_with_anyone_not_even_staff':'2191096_dcSzGJgJzR16j3ECEnO7lDrglEfCnQemNjhh54dYYjMqIa53VUlinqEPVO8i'}
def get_data():
    idItem = input("idItem:")
    idBot = input("idBot:")

    return idItem, idBot
    
def requestToken(idItem, idBot):
    # try:
    response = requests.post(
        url='https://csgoempire.com/api/v2/user/security/token',
        cookies = cok,
        headers = HEADERS,
        data= {
            "code": '126231',
            "uuid" : "b9928fe4-0994-4179-8718-523f467adf8f"
        }
    )
    print(response)
    if response.status_code == 200:
        print(response)
        # requestWithDraw(idItem, idBot, token)
    else:
        print("Failed")
    # except:
    #     print("Error")


def requestWithDraw(idItem, idBot, token):
    try:
        response = requests.post(
            url='https://csgoempire.com/api/v2/trade/withdraw',
            cookies = COOKIES,
            headers = HEADERS,
            data= {
                "item_ids": [idItem],
                "bot_id" : idBot,
                "security_token": token
            }
        )
        if response.status_code == 200:
            print("Succeed") 
        else:
            print("Failed")
    except Exception as e:
        print("Error", e)
    
    main()

def main():
    data = get_data()

if __name__ == "__main__":
    # main()
    # idItem, idBot = get_data()
    requestToken("1u136eu", "7868007")
