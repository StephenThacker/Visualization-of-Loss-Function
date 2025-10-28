import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from circles import readimagesintonumpyArray,maketrainingdata,makegroundtruth2, takecenter, custom_objects
Model1 = tf.keras.models.load_model('/home/stephen/Downloads/TF2UNET/unet/scripts/circles/2021-05-31T15-37_08/',custom_objects=custom_objects)
Model1.compile(loss='binary_crossentropy',run_eagerly=True)
Model2 = tf.keras.models.load_model('/home/stephen/Downloads/TF2UNET/unet/scripts/circles/2021-05-31T15-37_08/',custom_objects=custom_objects)
Model2.compile(loss='binary_crossentropy',run_eagerly = True)

Ground_truth = readimagesintonumpyArray('/home/stephen/Downloads/TF2UNET/unet/src/unet/datasets/Neuron/Ground/')[0]
Ground_truth = makegroundtruth2(Ground_truth)
Ground_truth = takecenter(Ground_truth)
train_data = readimagesintonumpyArray('/home/stephen/Downloads/TF2UNET/unet/src/unet/datasets/Neuron/Training/Training1/')[0]
train_data = maketrainingdata(train_data)
import matplotlib as plt

print(Model1.evaluate(train_data,Ground_truth))



#Layer1 = new_model.get_layer('conv_block')



# generates a multivariate normally distributed random vector with the same dimension as your tensorflow model
#Covariance Matrix of 1's was used.
def generateRandomVector(Model):
    A = []
    for elem in Model.weights:
        if 'kernel' in elem.name:
            temp = np.random.multivariate_normal(np.zeros(elem.shape[-1]), np.ones((elem.shape[-1], elem.shape[-1])),(elem.shape[0], elem.shape[1], elem.shape[2]))
            A += [temp]
    return A

#gets the (i,j)th Frobenius norm of the ith filter in the jth layer and stores it into a Matrix
def GetWeightsFrobeniusNorm(Model):
    A = []
    for elem in Model.weights:
        if 'kernel' in elem.name:
            B = []
            timmy = tf.transpose(elem,[3,2,0,1])
            for i in range(0,len(timmy)):
                temp = np.linalg.norm(np.ravel(timmy[i]))
                B += [temp]
            A += [B]

    return A

#Gets the Frobenius norm of the (i,j)th (layer,filter)
def GetVectorFrobeniusNorm(random_vector):
    A = []
    for elem in random_vector:
        B = []
        temp2 = np.swapaxes(elem,3,2)
        temp2 = np.swapaxes(temp2,2,1)
        temp2 = np.swapaxes(temp2,0,1)
        temp2 = np.swapaxes(temp2,3,2)
        temp2 = np.swapaxes(temp2,2,1)
        for i in range(0,len(temp2)):
            temp = np.linalg.norm(np.ravel(temp2[i]))
            B += [temp]
        A += [B]
    return A

#normalize a random vector according to the research paper's instructions
#The paper is Visualizing the Loss Landscape of Neural Nets by Li. The instructions can be found
# under section 4
def NormalizeVector(random_vector,Vector_Norms,Weight_Norms):
    j = 0
    for elem in random_vector:
        timmy = np.swapaxes(elem, 3, 2)
        timmy = np.swapaxes(timmy, 2, 1)
        timmy = np.swapaxes(timmy, 0, 1)
        timmy = np.swapaxes(timmy, 3, 2)
        timmy = np.swapaxes(timmy, 2, 1)
        for i in range(0,len(timmy)):
            timmy[i] = timmy[i]*(Weight_Norms[j][i]/Vector_Norms[j][i])
        normalized_random_vector = timmy
        normalized_random_vector = np.swapaxes(normalized_random_vector, 0, 1)
        normalized_random_vector = np.swapaxes(normalized_random_vector, 1, 2)
        normalized_random_vector = np.swapaxes(normalized_random_vector, 2, 3)
        elem = normalized_random_vector
        j += 1

    return random_vector


def addvectortotensorflowModel(Model,random_vector):
    i = 0
    for elem in Model.weights:
        if "kernel" in elem.name:
            temp = tf.math.add(random_vector[i], elem)
            elem.assign(temp)
            i += 1


#Computes a Matrix containing the contour plot information of the form [x_vector*alpha,y_vector*Beta, Loss Function Value]

def ComputeContourMatrix(Model, ModelCopy, x_range, y_range, Random_vector1, Random_vector2, Num_steps, training_inputs, training_groundtruths):
    Step_x = (2*x_range)/Num_steps
    Step_y = (2*y_range)/Num_steps

    x_start = -x_range
    y_start = -y_range

    print(Model.evaluate(training_inputs,training_groundtruths))


    A = []

    for i in range(0,Num_steps):
        B = []
        C = []
        for j in range(0,Num_steps):

            x_times_alpha_current = x_start + i*Step_x
            y_times_Beta_current = y_start + j*Step_y
            C += [y_times_Beta_current]

            print("random vector1")
            print(x_times_alpha_current)
            print("random vector2")
            print(y_times_Beta_current)


            Random_vector1 = [x*x_times_alpha_current for x in Random_vector1]
            Random_vector2 = [x*y_times_Beta_current for x in Random_vector2]

            addvectortotensorflowModel(Model,Random_vector1)
            addvectortotensorflowModel(Model,Random_vector2)

            B += [Model.evaluate(training_inputs,training_groundtruths, batch_size = 1)]


            k = 0

            for elem in Model.weights:
                if 'kernel' in elem.name:
                    elem.assign(ModelCopy.weights[k])

                k+= 1
        A += [B]

    return [A,C]




vector = generateRandomVector(Model1)
vector2 = generateRandomVector(Model1)
vector1 = vector
weights_norms = GetWeightsFrobeniusNorm(Model2)
Vector_Frobenius_norms = GetVectorFrobeniusNorm(vector)
normalized_vector1 = NormalizeVector(vector1,Vector_Frobenius_norms,weights_norms)
normalized_vector2 = NormalizeVector(vector2,Vector_Frobenius_norms,weights_norms)

Contour_Matrix =  ComputeContourMatrix(Model1, Model2 ,5,5,vector1,vector2, 100, train_data , Ground_truth)
X,Y = np.meshgrid(Contour_Matrix[1],Contour_Matrix[1])
plt.pyplot.contour(X,Y,Contour_Matrix[0])
