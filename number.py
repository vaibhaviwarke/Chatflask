test_string = "[3] police number in mumbai 7796384912"
  
# printing original string  
print("The original string : " + test_string) 
  
# using List comprehension + isdigit() +split() 
# getting numbers from string  
#res = [int(i) for i in test_string.split(" ") if i.isdigit()] 
for i in test_string.split():
		if i.isdigit():
			res=int(i)
a="+91"+str(res)
print("The numbers  is : "+str(a))
numb=int(a)
print(type(numb)) 


# print result 
#print("The numbers list is : " + str(res)) 