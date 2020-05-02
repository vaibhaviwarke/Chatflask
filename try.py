

'''from sklearn.feature_extraction.text import TfidfVectorizer


tokenized_list_of_sentences = [['this', 'is', 'one', 'basketball','baseball'], ['this', 'is', 'a', 'football']]

def identity_tokenizer(text):
    return text

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)    
x=tfidf.fit_transform(tokenized_list_of_sentences)
print(x)'''


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# The usual creation of arrays produces wrong format (as cosine_similarity works on matrices)
x = np.array([[0.577350,0.577350,0.577350]])
y = np.array([[0.5086718718935652,0.401042746469996,0.401042746469996]])

'''# Need to reshape these
x = x.reshape(1,-1)
y = y.reshape(1,-1)

# Or just create as a single row matrix
z = np.array([[1,1,1,1]])'''

# Now we can compute similarities
print(cosine_similarity(x,y)) # = array([[ 0.96362411]]), most similar
#cosine_similarity(x,z) # = array([[ 0.80178373]]), next most similar
#cosine_similarity(y,z) # = array([[ 0.69337525]]), least similar



	