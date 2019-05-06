
import urllib.parse
import urllib.request

"""
    Args:
    - isupper
    0 : No composed upper image
    1 : composed upper image
    - category
    1001 : men_tshirts - 0001 ~
    1002 : men_nambang - 1001 ~
    1003 : men_long	
    1101 : men_pants - 5001 ~
"""
data = urllib.parse.urlencode({
    'userid': 'test',
    'imageid' : '123456678',
    'upperid': '1539752779',
	'lowerid' : '005043',
    'isupper' : '0', 
    'category': '1101'
})


url = "http://127.0.0.1:10008/submit"
#url = "http://13.209.7.130:10008/submit"

data = data.encode('ascii')
with urllib.request.urlopen(url, data) as f:
    print(f.read().decode('utf-8'))
