import requests
 
url = "https://www.fast2sms.com/dev/bulk"
 
payload = "sender_id=FSTSMS&message=Emergency!!!!&language=english&route=p&numbers=9518331878,7030293724"

headers = {
 'authorization': "f1VtTkAPcLywzW7r23v08QbmoGaSYs9HDgNMpElJdxnCi5BeRqctE0TFHvSshruz5l6WfiY1RpwLD3Vy",
 'Content-Type': "application/x-www-form-urlencoded",
 'Cache-Control': "no-cache",
 }
 
response = requests.request("POST", url, data=payload, headers=headers)
 
print(response.text)