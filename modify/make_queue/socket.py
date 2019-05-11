import socket

s = socket.socket()
host = socket.gethostname()
port = 12222

s.connect((host, port))
print( 'Connected to', host)

send_str = str(userid) + " " + str(imageid) + " " + str(upperid) + " " + str(lowerid) + " " + str(isupper) + " " + str(category)
print(send_str)
print('##\n\n')
print(s.send(send_str.encode('utf-8')))

recv_str = (s.recv(1024)).decode('utf-8')
print(recv_str)
