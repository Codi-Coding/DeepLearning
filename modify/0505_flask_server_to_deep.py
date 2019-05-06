#-*- coding: utf-8 -*-


import threading

from flask import Flask, render_template, request, url_for
import time
import socket
import os

import logging
from logging import Formatter
from werkzeug import secure_filename
UPLOAD_FOLDER = '../testdata/'
PROD_FOLDER = '../data/test/'



################flask################
HOST_IP = "0.0.0.0"
host = socket.gethostname()
port = 12222
preprocess_port = 12224


app = Flask(__name__)
app.config.from_object(__name__)

app.config['LOGGING_LEVEL'] = logging.DEBUG 
app.config['LOGGING_FORMAT'] = '%(asctime)s %(levelname)s: %(message)s in %(filename)s:%(lineno)d]' 
app.config['LOGGING_LOCATION'] = './' 
app.config['LOGGING_FILENAME'] = 'abuse_detect.log' 
app.config['LOGGING_MAX_BYTES'] = 100000
app.config['LOGGING_BACKUP_COUNT'] = 1000 

def SockSend(userid,imageid,upperid,lowerid,isupper,category):
	s = socket.socket()
	host = socket.gethostname()
	port = 12222
	s.connect((host, port))
	print( 'Connected to', host)

	send_str = str(userid) + " " + str(imageid) + " " + str(upperid) + " " + str(lowerid) + " " + str(isupper) + " " + str(category)
	print(send_str)
	print('##\n\n')
	print(s.send(send_str.encode('utf-8')))
	print(s)
	recv_str = (s.recv(1024)).decode('utf-8')
	print("3")
	print(recv_str)

	return recv_str
	#s.close()


def SockSend_PreProcess(userid,imageid):
	s = socket.socket()
	host = socket.gethostname()
	port = 12224

	s.connect((host, port))
	print( 'Connected to', host)


  
	send_str = str(userid) + " " + str(imageid)
	print(send_str)
	print('##\n\n')
	print(s.send(send_str.encode('utf-8')))

	
	
def SockSend_ProdProcess(prodid,category):
	s = socket.socket()
	host = socket.gethostname()
	port = 12226

	s.connect((host, port))
	print( 'Connected to', host)

	send_str = str(prodid) + " " + str(category)
	print(send_str)
	print('##\n\n')
	print(s.send(send_str.encode('utf-8')))
	
	
	
	
#Define a route for url
@app.route('/')
def form():
	#imgplus()
	time.sleep(3)
	return "HI BRO"

#form action
@app.route('/submit', methods=['POST'] )
def action():
	try:
		#userid = request.args.get('userid',"")
		userid = request.form['userid']
		print("USER ID    : " + userid)
		#productid = request.args.get('productid',"")
		imageid = request.form['imageid']
		print("image id : " + imageid)
		upperid = request.form['upperid']
		print("Upper PRODUCT ID : " + upperid)
		lowerid = request.form['lowerid']
		print("Lower PRODUCT ID : " + lowerid)
		isupper = request.form['isupper']
		print("isupper : " + isupper)
		try:
			category = request.form['category']
		except:
			category = "ERRORGORY"
		print("CATEGORY   : " + category)
		if len(userid) > 3 and len(upperid) > 3 and len(lowerid) > 3:
			send_str = str(userid) + " " + str(imageid) + " " + str(upperid) + " " + str(lowerid) + " " + str(isupper) + " " + str(category)
			print(send_str)
			#try:
			print("SOCKSEND")
			data = SockSend(userid,imageid,upperid,lowerid,isupper,category)

			return data
			#catch:
			#print("연결 실패")
		else:
			print("wrong data")
		print("\n\n")
	except:
		"submit ERROR"
	'''
	s = socket.socket()
	host = socket.gethostname()
	port = 12222

	s.connect((host, port))
	print( 'Connected to', host)

	#z = input("Enter something for the server: ")
	#s.send(z.encode('utf-8'))


	print(s)
	send_str = userid + " " + productid
	print(s.send(send_str.encode('utf-8')))
	s.close()
	'''
	time.sleep(3)
	return "http://spnimage.edaily.co.kr/images/photo/files/NP/S/2017/02/PS17022200199.jpg"

#form action
@app.route('/upload', methods=['POST'] )
def uploadaction():
	productid = request.form['productid']
	print(productid)
	#time.sleep(3)
	return "Success"


@app.route('/useruploadview', methods=['get'] )
def uploadview():
	f = open("web_debug.txt","r")
	data = f.read()
	return '''
		
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
	<p><input type=file name=userpicture>
	 <input type=submit value=Upload><br><br>
	 userid : <input type=text name = userid>
	 imageid : <input type=text name = imageid>
	</form>''' + data
	
#upload user picture
@app.route('/userupload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		try: 
			userid = request.form['userid']
			print("USER ID    : " + userid)
			#productid = request.args.get('productid',"")
			imageid = request.form['imageid'] 
			print("image id : " + imageid)

			upload_path = UPLOAD_FOLDER + "/" + userid + "/input/body_images/"

			file = request.files['userpicture']
			filename = secure_filename(file.filename)
			#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			file.save(os.path.join(upload_path, imageid+ "_0.jpg"))

			SockSend_PreProcess(userid,imageid)
			#print(url_for('uploaded_file',filename=filename))
			return "OK"#redirect(url_for('uploaded_file',filename=filename))3

		except Exception as ex:
			print("ERRROR : ",ex)


	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
	<p><input type=file name=userpicture>
	 <input type=submit value=Upload><br><br>
	 userid : <input type=text name = userid>
	 imageid : <input type=text name = imageid>
	</form>'''


	
#upload user picture
@app.route('/productupload', methods=['GET', 'POST'])
def prod_upload_file():
	if request.method == 'POST':
		#try: 
		prodid = request.form['prodid']
		print("PROD ID    : " + prodid)
		#productid = request.args.get('productid',"")
		category = request.form['category'] 
		print("category : " + category)
		
		if category == "1001":
			upload_path = PROD_FOLDER  + "men_tshirts/images"
		elif category == "1002":
			upload_path = PROD_FOLDER  + "men_nambang/images"
		elif category == " 1003":
			upload_path = PROD_FOLDER  + "men_long/images"
		elif category == " 1101":
			upload_path = PROD_FOLDER  + "men_pants/images"
			
		file = request.files['prodpicture']
		filename = secure_filename(file.filename)
		#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file.save(os.path.join(upload_path, prodid+ "_1.png"))
		#return "OKK " + prodid + " " + category 
		SockSend_ProdProcess(prodid,category)
		#print(url_for('uploaded_file',filename=filename))
		return "OK"#redirect(url_for('uploaded_file',filename=filename))3

		#except Exception as ex:
		#	print("ERRROR : ",ex)
		#	return "ERROR"


	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
	<p><input type=file name=prodpicture>
	 <input type=submit value=Upload><br><br>
	 prodid : <input type=text name = prodid>
	 category : <input type=text name = category>
	</form>'''
	
	
#Run the app
if __name__ == '__main__':
	#th = threading.Thread(target=imgplus,args=())
	app.run(host=HOST_IP, debug=True, port =10008)
