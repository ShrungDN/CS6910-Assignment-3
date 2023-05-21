import pickle
import matplotlib.pyplot as plt
with open('D:\IITM Academic Stuff\Sem 8 Books\CS6910\CS6910-Assignment-3\\attentions(1)', 'rb') as file:
    att = pickle.load(file)
plt.imshow(att)
plt.show()