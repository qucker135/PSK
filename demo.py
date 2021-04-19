with open("zgoda.png", "rb") as image:
	f = image.read()
	b = bytearray(f)
	for i in b:
		print(type(i))
	f = open('zgoda2.png', 'wb')
	f.write(b)
	f.close()	
