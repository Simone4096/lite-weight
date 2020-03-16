import numpy as np
import random
import copy
import csv

'''
----------NN------------

This class is our network, cointaining all the layers.

Requires a list with the numbers of neurons in each layer and a list of two
Activation objects (the first one for all the layers except the output one
and the second for the output) when creating an instance.


Attributes:
    
    n_layer -- number of layers, including input and output
    
    layer_arr -- list containing all the layers (Layer objects)
    
Methods:
    
    feedforward -- given the input data as a list, returns the output of
                   each layer as a list
                   
    backpropagation -- given the input data as a list, the target as a list
                       and the loss function, returns a Delta object
                       containing the gradient for each weight and bias in
                       each layer
                       
    update -- given a np matrix input, a np matrix target, the values of
              eta, lambda and alpha a Delta object dE_prec (for the momentum)
              and a loss function, updates the weights and biases in each
              layer and returns the actual variation of the weights and
              biases as a Delta object 
              
    update_zero -- same as update, but without dE_prec as an input
                       

'''

class NN:
    
    def __init__(self, vec_layer, act_func):
        
        self.n_layer = len(vec_layer)
        
        self.layer_arr = []
        
        for i in range(0, self.n_layer - 2):
            #creates a Layer object and appends it to layer_arr
            temp_layer = Layer(vec_layer[i], vec_layer[i + 1], act_func[0])
            self.layer_arr.append(temp_layer)
        
        #creates the last Layer object and appends it to layer_arr
        temp_layer = Layer(vec_layer[self.n_layer - 2], vec_layer[self.n_layer - 1],\
                           act_func[1])
        self.layer_arr.append(temp_layer)
            
            
    def feedforward(self, input_data):
        
        output_arr = [input_data]
        
        for i in range(0, self.n_layer - 1):
            #produces output of a layer and appends it to output_arr
            temp_output = self.layer_arr[i].act(output_arr[i])
            output_arr.append(temp_output)
            
        return output_arr

    
    def backpropagation(self, input_data, target, loss_f):
        
        output_arr = self.feedforward(input_data)
        actprime_arr = []
        
        for i in range(0, self.n_layer - 1):
            #calculates output with the derivative of the activation function
            #in each layer
            temp_actprime = self.layer_arr[i].actprime(output_arr[i])
            actprime_arr.append(temp_actprime)
            
        #differentiates between different loss functions
        if loss_f == MEE:
            den = 0
            
            for i in range(len(target)): 
                den += (output_arr[self.n_layer - 1][i] - target[i]) ** 2
                
            den = np.sqrt(den)
            
            #initializes first value of past_arr (see below)
            past_arr = [(1. / den) * (output_arr[self.n_layer - 1] - target)]
            
        if loss_f == MSE:
            #initializes first value of past_arr (see below)
            past_arr = [2 * (output_arr[self.n_layer - 1] - target)]
        
        for i in range(self.n_layer - 2, 0, -1):
            #calculates iteratively the derivative of the loss function with respect to
            #the output in each neuron
            temp_past = self.layer_arr[i].past(past_arr[self.n_layer - 2 - i], \
                        actprime_arr[i])
            past_arr.append(temp_past)
            
        #reverse order of past_arr
        past_arr = np.flipud(past_arr)
        
        dE_dw = []
        dE_db = []
        
        for i in range(0, self.n_layer - 1):
            #calculates the derivative of the loss function with respect to
            #the weight
            temp_dE = np.outer(past_arr[i] * actprime_arr[i], output_arr[i])
            dE_dw.append(temp_dE)
            
            #calculates the derivative of the loss function with respect to
            #the bias
            temp_dE = past_arr[i] * actprime_arr[i]
            dE_db.append(temp_dE)
            
        dE = Delta(np.array(dE_dw), np.array(dE_db))
        
        return dE
    
    
    def update_zero(self, input_arr, target_arr, eta, lamda, loss_f):
        n_batch = len(input_arr)
        
        dE = self.backpropagation(input_arr[0], target_arr[0], loss_f)
        
        #average of dE on a mini-batch
        for i in range(1, n_batch):
            temp_dE = self.backpropagation(input_arr[i], target_arr[i], loss_f)
            dE.dw += temp_dE.dw
            dE.db += temp_dE.db
            
        dE.dw = dE.dw / n_batch
        dE.db = dE.db / n_batch
        
        #copy to return for momentum
        dE_tot = copy.deepcopy(dE)
        
        #update all the weights and biases
        for i in range(0, self.n_layer - 1):
            dE_tot.dw[i] = (-1 * eta * dE.dw[i]) - (2 * lamda * self.layer_arr[i].w)
            dE_tot.db[i] = (-1 * eta * dE.db[i])
        
        for i in range(0, self.n_layer - 1):
            self.layer_arr[i].w += dE_tot.dw[i]
            self.layer_arr[i].b += dE_tot.db[i]
            
        return dE_tot
    
    
    def update(self, input_arr, target_arr, eta, lamda, alpha, dE_prec, loss_f):
        n_batch = len(input_arr)
        
        dE = self.backpropagation(input_arr[0], target_arr[0], loss_f)
        
        #average of dE on a mini-batch
        for i in range(1, n_batch):
            temp_dE = self.backpropagation(input_arr[i], target_arr[i], loss_f)
            dE.dw += temp_dE.dw
            dE.db += temp_dE.db
            
        dE.dw = dE.dw / n_batch
        dE.db = dE.db / n_batch
        
        #copy to return for momentum
        dE_tot = copy.deepcopy(dE)
        
        #update all the weights and biases
        for i in range(0, self.n_layer - 1):
            dE_tot.dw[i] = (-1 * eta * dE.dw[i]) - \
            (2 * lamda * self.layer_arr[i].w) + alpha * dE_prec.dw[i]
            dE_tot.db[i] = (-1 * eta * dE.db[i])
                                   
        for i in range(0, self.n_layer - 1):
            self.layer_arr[i].w += dE_tot.dw[i]
            self.layer_arr[i].b += dE_tot.db[i]
                                       
        return dE_tot
        

'''
----------Layer------------

This class is the ensamble of the set of weights between the same two layers
and the biases.

Requires the number of neuron to the left, the number of neuron to the right
and the activation function when creating an instance.


Attributes:
    
    w -- np matrix of weights
    
    b -- np array of biases
    
    act_func -- activation functions
    
Methods:
    
    act -- given the output of the preceding layer as a list, returns the
           output of current layer as a list
                   
    actprime -- given the output of the preceding layer as a list, returns
                the output of the current layer but using the derivative
                of the activation function instead of using the actual
                activation function
                       
    past -- calculates iteratively the past term (as defined above) 
                       

'''
    
class Layer:
    def __init__(self, sx, dx, act_func):
        #random initialization
        self.w = 1.4 * np.random.rand(dx, sx) - 0.7
        self.b = 1.4 * np.random.rand(dx) - 0.7

        self.act_func = act_func
    
    def act(self, prev_out):
        #matrix multiplication
        out = np.dot(self.w, prev_out) + self.b
        return self.act_func.func(out)
    
    def actprime(self, prev_out):
        #matrix multiplication
        out = np.dot(self.w, prev_out) + self.b
        return self.act_func.der(out)
    
    def past(self, right_past, right_actprime):
        var = right_past * right_actprime
        return np.dot(var, self.w)


'''
----------Delta------------

This class is the set of variation of weights and biases


Attributes:
    
    dw -- np matrix of variation of weights
    
    db -- np array of variation of biases
                       

'''

class Delta:
    def __init__(self, dw, db):
        self.dw = dw
        self.db = db    
    
    
'''
----------Activation------------

This class is the set of activation function and its derivative


Attributes:
    
    func -- activation function
    
    der -- derivative of activation function
                       

'''
    
class Activation:
    def __init__(self, func, der):
        self.func = func
        self.der = der
        

'''
----------Data------------

This class is the set of input and target matrices


Attributes:
    
    inp -- input matrix
    
    tar -- target array
                       

'''
        
class Data:
    def __init__(self, inp, tar):
        self.inp = inp
        self.tar = tar
        
        
'''
----------Hyper------------

This class is the set of hyperparameters


Attributes:
    
    eta -- learning rate
    
    lamda -- regularization constant
    
    alpha -- momentum constant
    
    k_hl -- vector hidden layer(s)
    
    n_batch -- lenght of mini-batch
    
    score -- non-normalized accuracy
    
    score_rec -- record of non-normalized accuracy
    
    loss -- MSE or MEE on test (or validation)
    
    loss_rec -- record of MSE or MEE
    
    loss_train -- MSE or MEE on train
                       

'''

class Hyper:
    def __init__(self, eta, lamda, alpha, k_hl, n_batch):
        self.eta = eta
        self.lamda = lamda
        self.alpha = alpha
        self.k_hl = k_hl
        self.n_batch = n_batch
        self.score = 0
        self.score_rec = []
        self.loss = 0
        self.loss_rec = []
        self.loss_train = 0


#sigmoid   
def sigmoid_fun(x):
    val = 1. / (1. + np.exp(-x)) 
    return val
    
#sigmoid derivative
def sigmoid_der(x):
    val = np.exp(-x) / ((np.exp(-x) + 1.) * (np.exp(-x) + 1.))
    return val

#reLU
def relu_fun(x):
    val = 0
    if x > 0:
        val = x
    return val

#reLU derivative
def relu_der(x):
    val = 0
    if x > 0:
        val = 1
    return val

#hyperbolic tangent
def tanh_fun(x):
    val = np.tanh(x)
    return val

#hyperbolic tangent derivative
def tanh_der(x):
    val = 1. / (np.cosh(x) * np.cosh(x))
    return val


def identity_fun(x):
    return x


def identity_der(x):
    return 1

#given out and target arrays of lenghts two, returns 1 if they match and
#0 otherwise
def bin_compare(out, target):
    #calculates the index of the max
    ind = np.argmax(out)
    res = (1 - ind) * target[0] + ind * target[1]
    return res

#Mean Square Error between out and target arrays
def MSE(out, target):
    lenght = len(target)
    
    res = 0
    
    for i in range(lenght):
        res += (out[i] - target[i]) * (out[i] - target[i])
        
    return res


#Mean Euclidean Error between out and target arrays
def MEE(out, target):
    lenght = len(target)
    
    res = 0
    
    for i in range(lenght):
        res += (out[i] - target[i]) * (out[i] - target[i])
        
    res = np.sqrt(res)
        
    return res


#loads monk data and preprocesses them with 1-of-k encoding
#returns a Data object
def load_monk(loc):
    with open(loc) as f:
        data = f.read()

    data = data.split('\n')
    data = [item[1:len(item)] for item in data if item != '']

    mat = [[int(row.split(' ')[int(i)]) for row in data] for i in range(0,7)]
    
    mat2 = []
    
    #1-of-k encoding
    for j in range(len(mat[0])):
        row_temp = np.array([])
        
        for i in range(len(mat)):
            if i == 0:
                temp = np.array([int(mat[i][j] == 0), int(mat[i][j] == 1)])
                row_temp = np.append(row_temp, temp)
                
            if i in [1, 2, 4]:
                temp = np.array([int(mat[i][j] == 1), int(mat[i][j] == 2), int(mat[i][j] == 3)])
                row_temp = np.append(row_temp, temp)
                
            if i in [3, 6]:
                temp = np.array([int(mat[i][j] == 1), int(mat[i][j] == 2)])
                row_temp = np.append(row_temp, temp)
                
            if i == 5:
                temp = np.array([int(mat[i][j] == 1), int(mat[i][j] == 2), int(mat[i][j] == 3), int(mat[i][j] == 4)])
                row_temp = np.append(row_temp, temp)
            
        mat2.append(row_temp)
    
    mat2 = np.array(mat2)
    
    target = mat2[:, :2]
    
    data = mat2[:, 2:]
    
    res = Data(data, target)
    
    return res



#loads cup data already splitted in internal test and training
#returns a Data object
def load_train_cup(loc):
    
    with open(loc) as infile:
        reader = csv.reader(infile, delimiter=",")
        
        inputh = []
        target = []
        
        for row in reader:
            tar = np.array([float(row[11]), float(row[12])])
            inp = np.array([float(x) for x in row[1:11]])
            target.append(tar)
            inputh.append(inp)
            
    inputh = np.array(inputh)
    target = np.array(target)

    res = Data(inputh, target)
    
    return res


#creates a batch of lenght len_batch given a Data object
#returns a Data object
def create_batch(len_batch, dataset):
    
    num_i = random.sample(range(len(dataset.tar)), len_batch)
    
    new_data_train = []
    new_target_train = []
    
    for val in num_i:
        new_data_train.append(dataset.inp[val])
        new_target_train.append(dataset.tar[val])
        
    batch = Data(np.array(new_data_train), np.array(new_target_train))
 
    return batch
    

#cross validation function
#takes the grid of hyperparameters, the train set, the sigma Activation object
#the lenght of input and output layers, the loss function and the task (monk or
#cup)
#returns a list of explored hyperparameters (Hyper objects) for every subset
#of the training togheter with the loss and score
def cross_val(range_hyp, num_fold, train_set, sigma, inp_out_lay, epochs, loss_f, task):
    
    #shuffle training set
    train_shuffled = create_batch(len(train_set.tar), train_set)
    
    k_fold = []
    
    hyper_fold = []
    
    #divides training set in num_fold parts
    for i in range(num_fold):
        
        inp_temp = np.array([train_shuffled.inp[k] for k in range(0,len(train_set.tar)) \
                             if k % num_fold == i])
        tar_temp = np.array([train_shuffled.tar[k] for k in range(0,len(train_set.tar)) \
                             if k % num_fold == i])
        
        data_temp = Data(inp_temp, tar_temp)
        
        k_fold.append(data_temp)
        
    
    for i in range(num_fold):
        
        hyper_fixed_fold = []
        
        validation = copy.deepcopy(k_fold[i])
        
        i_next = (i + 1) % num_fold
        
        #validation set
        train_validation = copy.deepcopy(k_fold[i_next])
        
        #merging of data used as training 
        for j in range(i + 2, i + num_fold):
            j_mod = j % num_fold
            train_validation.inp = np.concatenate((train_validation.inp, k_fold[j_mod].inp))
            train_validation.tar = np.concatenate((train_validation.tar, k_fold[j_mod].tar))
        
        #explore grid
        for eta in range_hyp.eta:
            for lamda in range_hyp.lamda:
                for alpha in range_hyp.alpha:
                    for k_hl in range_hyp.k_hl:
                        for n_batch in range_hyp.n_batch:
                            
                            layers = np.append(np.append(inp_out_lay[0], k_hl), inp_out_lay[1])

                            #creates a NN object
                            nn = NN(layers, sigma)
                            
                            #updates nn with update_zero
                            batch0 = create_batch(n_batch, train_validation)
                            dE_temp = nn.update_zero(batch0.inp, batch0.tar, eta, lamda, loss_f)
                            
                            tot_init = 0
                            loss_init = 0
                            
                            #calculates first score and loss with MSE or MEE
                            for j in range(0, len(validation.tar)):
                                output = nn.feedforward(validation.inp[j])[nn.n_layer - 1]
                                target = validation.tar[j]
                                tot_init += bin_compare(output, target)
                                if loss_f == MEE:
                                    loss_init += MEE(output, target)
                                if loss_f == MSE:
                                    loss_init += MSE(output, target)
                            
                            tot_arr = [tot_init]
                            loss_arr = [loss_init]
                            
                            if task == "monk":
                                new = 0
                                stop = 1000
                                avg = 200
                                
                            if task == "cup":
                                new = 1000000
                                stop = 4000
                                avg = 400
                            
                            k_max = epochs - 1
                            
                            #loop for training
                            for k in range(1, epochs):
                                batch = create_batch(n_batch, train_validation)
                                #updates nn with update
                                dE_temp = nn.update(batch.inp, batch.tar, eta, lamda,\
                                                    alpha, dE_temp, loss_f)
                                
                                temp_tot = 0
                                temp_loss = 0
                                
                                #calculates score and loss with MSE or MEE
                                for j in range(0, len(validation.tar)):
                                    output = nn.feedforward(validation.inp[j])[nn.n_layer - 1]
                                    target = validation.tar[j]
                                    temp_tot += bin_compare(output, target)
                                    if loss_f == MEE:
                                        temp_loss += MEE(output, target)
                                    if loss_f == MSE:
                                        temp_loss += MSE(output, target)
                                    
                                tot_arr.append(temp_tot)
                                loss_arr.append(temp_loss)
                                
                                
                                
                                #dynamic stop condition
                                if k % stop == 0:
                                    
                                    old = new
                                    new = 0
                                    
                                    #averaging of score or loss
                                    for j in range(avg):
                                        if task == "cup":
                                            new += loss_arr[k - j]
                                        if task == "monk":
                                            new += tot_arr[k - j]
                                    
                                    delta = new - old
                                    
                                    #exit condition
                                    if task == "cup":
                                        if delta >= 0:
                                            k_max = k
                                            break
                                        
                                    if task == "monk":
                                        if delta <= 0:
                                            k_max = k
                                            break
                                        
                                        
                            if (k_max > stop and k_max != (epochs - 1)):
                                k_max -= stop
                                
                            hyperpar = Hyper(eta, lamda, alpha, k_hl, n_batch)
                            
                            score = tot_arr[k_max]
                            
                            #normalizing the MSE or MEE
                            loss_val = loss_arr[k_max] / len(validation.tar)
                            
                            if task == "monk":
                                hyperpar.score = score
                                hyperpar.score_rec = tot_arr
                            
                            hyperpar.loss = loss_val
                            hyperpar.loss_rec = loss_arr
                                
                            
                            loss_train = 0
                            
                            for j in range(0, len(train_validation.tar)):
                                output = nn.feedforward(train_validation.inp[j])[nn.n_layer - 1]
                                target = train_validation.tar[j]
                                if loss_f == MEE:
                                    loss_train += MEE(output, target)
                                if loss_f == MSE:
                                    loss_train += MSE(output, target)
                            
                            #normalizing the MSE or MEE
                            loss_train = loss_train / len(train_validation.tar)
                            
                            hyperpar.loss_train = loss_train
                            
                            hyper_fixed_fold.append(hyperpar)
                            
                            #printing results
                            if task == "monk":
                                print("Accuracy validation: ", score / len(validation.tar), "\n")
                                hyperpar.score = score
                                hyperpar.score_rec = tot_arr
                                
                            print("Loss train: ", loss_train, "Loss validation: ", loss_val, "\n")
                            print("N. fold: ", i, "\n")
                            print("Eta: ", eta, " Lambda: ", lamda, " Alpha: ", alpha, "\n")
                            print("Neuroni hidden layer: ", k_hl, "\n")
                            print("Lunghezza batch: ",n_batch, ' Stop point : ', k_max, "\n")
                            print("\n")
                            
    
        hyper_fold.append(hyper_fixed_fold)

        
    return hyper_fold

#demonstration function for monk sets
#it requires the path of test and training set, and the monkset number
def start_monk(loc_train, loc_test, monk_num):
    
    train_set = load_monk(loc_train)
    test_set = load_monk(loc_test)
    
    sigma1 = Activation(sigmoid_fun, sigmoid_der)
    
    sigma = [sigma1, sigma1]
    
    loss_f = MSE

    task = "monk"
    
    layers = np.array([17,2])
    
    max_val = 10000
    n_fold = 5
    
    print("Inizio Cross Validation \n")
    print("\n")
    
    #fixed grid of hyperparameters
    range_hyp_1 = Hyper([0.3, 0.4], [0.00001], [0.45, 0.9], [[5], [10]], [10])
    range_hyp_2 = Hyper([0.45, 0.8], [0.00001, 0.001], [0.9], [[3], [15]], [30])
    range_hyp_3 = Hyper([0.2], [0.0001, 0.0], [0.0], [[10], [30]], [30])
    
    if monk_num == 1:
        hyper_matrix = cross_val(range_hyp_1, n_fold, train_set, sigma, layers, max_val, loss_f, task)
        
    if monk_num == 2:
        hyper_matrix = cross_val(range_hyp_2, n_fold, train_set, sigma, layers, max_val, loss_f, task)
        
    if monk_num == 3:
        hyper_matrix = cross_val(range_hyp_3, n_fold, train_set, sigma, layers, max_val, loss_f, task)

    print("Fine Cross Validation \n")
    print("\n")
    
    print("Inizio Training e Testing \n")
    print("Iperparametri: \n")
    
    #fixed hyperparameters for training and testing in each monkset
    if monk_num == 1:
        print("eta: 0.4, lambda = 0.00001, alpha = 0.9, neuroni hidden layer = 10, lunghezza batch = 10 \n")
        stop = 2000
        eta = 0.4
        lamda = 0.00001
        alpha = 0.9
        layer = [17, 10, 2]
        n_batch = 10
        
    if monk_num == 2:
        print("eta: 0.8, lambda = 0.00001, alpha = 0.9, neuroni hidden layer = 15, lunghezza batch = 30 \n")
        stop = 2000
        eta = 0.8
        lamda = 0.00001
        alpha = 0.9
        layer = [17, 15, 2]
        n_batch = 30
        
    if monk_num == 3:
        print("eta: 0.2, lambda = 0.0001, alpha = 0.0, neuroni hidden layer = 30, lunghezza batch = 30 \n")
        stop = 1000
        eta = 0.2
        lamda = 0.0001
        alpha = 0.0
        layer = [17, 30, 2]
        n_batch = 30
        
    epochs = stop + 1
    

    #creates a NN object                            
    monk = NN(layer, sigma)
        
    #updates monk with update_zero
    batch0 = create_batch(n_batch, train_set)
    dE_temp = monk.update_zero(batch0.inp, batch0.tar, eta, lamda, loss_f)
    
    
    loss_init = 0
    loss_init_train = 0
    tot_init = 0
    tot_init_train = 0
    
    #calculating initial loss and score in each of these cycles
    for j in range(0, len(test_set.tar)):
        output = monk.feedforward(test_set.inp[j])[monk.n_layer - 1]
        target = test_set.tar[j]
        loss_init += MSE(output, target)
        tot_init += bin_compare(output, target)
        
    for j in range(0, len(train_set.tar)):    
        output = monk.feedforward(train_set.inp[j])[monk.n_layer - 1]
        target = train_set.tar[j]
        loss_init_train += MSE(output, target)
        tot_init_train += bin_compare(output, target)
    
    loss_arr = [loss_init]
    loss_arr_train = [loss_init_train]
    
    tot_arr = [tot_init]
    tot_arr_train = [tot_init_train]
    
    
    for k in range(1, epochs):
        #updates monk with update
        batch = create_batch(n_batch, train_set)
        dE_temp = monk.update(batch.inp, batch.tar, eta, lamda, alpha, dE_temp, loss_f)
        
        temp_loss = 0
        temp_loss_train = 0
        
        temp_tot = 0
        temp_tot_train = 0
        
        #calculating loss and score after each update
        for j in range(0, len(test_set.tar)):
            output = monk.feedforward(test_set.inp[j])[monk.n_layer - 1]
            target = test_set.tar[j]
            temp_loss += MSE(output, target)
            temp_tot += bin_compare(output, target)
        
        for j in range(0, len(train_set.tar)): 
            output = monk.feedforward(train_set.inp[j])[monk.n_layer - 1]
            target = train_set.tar[j]
            temp_loss_train += MSE(output, target)
            temp_tot_train += bin_compare(output, target)
    
        loss_arr.append(temp_loss)
        loss_arr_train.append(temp_loss_train)
        
        tot_arr.append(temp_tot)
        tot_arr_train.append(temp_tot_train)
            
    #renormalizing loss
    loss_arr = np.array(loss_arr) / len(test_set.tar)
    loss_arr_train = np.array(loss_arr_train) / len(train_set.tar)
    
    #calculating the actual loss and score in the stopping point
    loss_train = loss_arr_train[stop]
    loss_val = loss_arr[stop]
    
    score = tot_arr[stop]
    score_train = tot_arr_train[stop]
    
    
    print("MSE test: ", loss_val)
    print("Accuracy test: ", float(score) * 100 / len(test_set.tar), "%")
    
    print("MSE train: ", loss_train)
    print("Accuracy train: ", float(score_train) * 100 / len(train_set.tar), "%")
    
    

        
#demonstration function for cup sets
#requires the path of internal test and training set  
def start_cup(loc_train, loc_int_test):
    
    train_set = load_train_cup(loc_train)
    test_set = load_train_cup(loc_int_test)
    
    sigma1 = Activation(sigmoid_fun, sigmoid_der)
    sigma2 = Activation(identity_fun, identity_der)
    
    sigma = [sigma1, sigma2]
    
    loss_f = MEE
    
    task = "cup"
    
    layers = np.array([10,2])
    
    print("Inizio Cross Validation \n")
    print("\n")
    
    range_hyp = Hyper([0.01, 0.05], [0.00001, 0.000001], [0], [[30]], [50])
    
    max_val = 50000
    n_fold = 5
    
    #fixed grid of hyperparameters
    hyper_matrix = cross_val(range_hyp, n_fold, train_set, sigma, layers, max_val, loss_f, task)

    print("Fine Cross Validation \n")
    print("\n")
    
    print("Inizio Training e Testing \n")
    print("Iperparametri: \n")
    
    print("eta: 0.05, lambda = 0.000001, alpha = 0.0, neuroni hidden layer = 30, lunghezza batch = 50 \n")
        
    #creates a NN object
    cup = NN([10, 30, 2], sigma)
    
    #fixed hyperparameters for training and testing
    n_batch = 50
    eta = 0.05
    lamda = 0.000001
    alpha = 0
    epochs = 22501
    
    stop_point = 22500
    
    #updates cup with update_zero
    batch0 = create_batch(n_batch, train_set)
    dE_temp = cup.update_zero(batch0.inp, batch0.tar, eta, lamda, loss_f)
    
    
    loss_init = 0
    loss_init_train = 0
    
    #calculating initial loss
    for j in range(0, len(test_set.tar)):
        output = cup.feedforward(test_set.inp[j])[cup.n_layer - 1]
        target = test_set.tar[j]
        loss_init += MEE(output, target)
        
    for j in range(0, len(train_set.tar)):    
        output = cup.feedforward(train_set.inp[j])[cup.n_layer - 1]
        target = train_set.tar[j]
        loss_init_train += MEE(output, target)
    
    loss_arr = [loss_init]
    loss_arr_train = [loss_init_train]
    
    
    for k in range(1, epochs):
        #updates cup with update
        batch = create_batch(n_batch, train_set)
        dE_temp = cup.update(batch.inp, batch.tar, eta, lamda, alpha, dE_temp, loss_f)
        
        temp_loss = 0
        temp_loss_train = 0
        
        #calculating loss after each update
        for j in range(0, len(test_set.tar)):
            output = cup.feedforward(test_set.inp[j])[cup.n_layer - 1]
            target = test_set.tar[j]
            temp_loss += MEE(output, target)
        
        for j in range(0, len(train_set.tar)): 
            output = cup.feedforward(train_set.inp[j])[cup.n_layer - 1]
            target = train_set.tar[j]
            temp_loss_train += MEE(output, target)
    
        loss_arr.append(temp_loss)
        loss_arr_train.append(temp_loss_train)
            
    #renormalizing loss
    loss_arr = np.array(loss_arr) / len(test_set.tar)
    loss_arr_train = np.array(loss_arr_train) / len(train_set.tar)
    
    #calculating the actual loss and score in the stopping point
    loss_train = loss_arr_train[stop_point]
    loss_val = loss_arr[stop_point]
    
    
    print("Loss: ", loss_val)
    print("Loss train: ", loss_train)




#random seed initialization
np.random.seed()
random.seed()