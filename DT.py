import sys
import csv
import numpy as np


# getdata() function definition
def getdata(fileName):
    array = []
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            spliting=row[0].split()
            array.append([]) #[[]]
            array[-1]=spliting
    attributes_Array=array[0][1:]
    instances_Array=array
    return attributes_Array,instances_Array


def impurity(m,n):# Truelist/Falselist: m is number of instances(no. of rows) in class A, n is number of instances in class B
    aveg_imp = (m * n / np.square(m+n))
    return aveg_imp

def is_pure(ins):
    class1=0
    class2=0
    for i in range(1,len(ins)):#skip first row
        if ins[i][0]=="live":
            class1 +=1
        else:
            class2 +=1
    if class1==0:
        return "die"
    elif class2==0:
        return "live"
    else:
        return "impure"

def cont_instaces(inst):
    count_live = 0
    count_die = 0
    for k in range(1, len(inst)):#skip first row
        if inst[k][0] == 'live':
            count_live += 1
        else:
            count_die += 1
    return count_live ,count_die

# instances: a list of nodes [['live', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'true', 'false', 'true', 'true', 'true', 'true', 'true', 'true', 'false'], NODE2,...]
# attributes: a list of attributes ['AGE', 'FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'BIGLIVER', 'FIRMLIVER', 'SPLEENPALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'SGOT', 'HISTOLOGY']

class Node:
    def __init__(self, attribute, left, right):
        self.attribute = attribute
        self.left = left
        self.right = right

    def print_Out(self, indent):#DFS
        print("{}{}=True".format(indent,self.attribute))
        self.left.print_Out(indent + "    ")
        print("{}{}=False".format(indent,self.attribute))
        self.right.print_Out(indent + "    ")

class leaf:
    def __init__(self, Class, prob):
        self.Class = Class
        self.prob=prob

    def print_Out(self , indent):
        print("{} Class: {} probability: {:.2f}".format(indent , self.Class,self.prob))

def get_classes(instance):
    class1 = 0
    class2 = 0
    for i in range(1,len(instance)):
        if instance[i][0] == "live":
            class1 += 1
        else:
            class2 += 1
    if class1 > class2:
        return "live" , class1/(class1+class2) #most common class, class probability
    else:
        return "die" , class2/(class1+class2)


# node.attribute = 'AGE'
# instance =['live', 'false', 'false', 'false', 'true', 'false', 'false', 'false', 'true', 'false', 'true', 'true', 'true', 'true', 'true', 'true', 'false']
def Classify(node,instance):
    if isinstance(node, leaf):
        return node.Class
    node_index = ['Class','AGE', 'FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'BIGLIVER', 'FIRMLIVER', 'SPLEENPALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'SGOT', 'HISTOLOGY'].index(node.attribute)
    if instance[node_index]=='true':
        return Classify(node.left,instance)
    else:
        return Classify(node.right, instance)

def BuildTree (instances, attributes):
    if len(instances)==1:
        Class , prob = get_classes(insdata)
        return leaf(Class , prob)
    if is_pure(instances)== "live" or is_pure(instances)=="die":
        return leaf(instances[1][0],1)
    if len(attributes)==0:
        Class , prob = get_classes(instances)
        return leaf(Class , prob)

    else :
        Weighted_avr=[]
        best_splits_true=[]#Store all trueList and falseList in arrays (same length as Weighted_avr)
        best_splits_false=[]
        for i in range(len(attributes)): # attributes[i]
            trueList =[instances[0]]
            falseList = [instances[0]]

            # find the index of attributes[i](string) in instance[0](list)
            index = instances[0].index(attributes[i])
            for j in range(1, len(instances)):#skip first row
                if instances[j][index] == "true":
                    trueList.append(instances[j])
                else:
                    falseList.append(instances[j])
            m, n = cont_instaces(trueList)
            if m == 0 and n ==0:# avoid choose attributes that has not splite the instances
                Weighted_avr.append(100000)
            else:
                true_impurity=impurity(m,n)
                total_nodeT = m + n
                m,n = cont_instaces(falseList)
                if m == 0 and n == 0:
                    Weighted_avr.append(100000)
                else:
                    false_impurity=impurity(m,n)
                    total_nodeF = m + n
                    Total_Node = total_nodeT + total_nodeF
                    Pro_nodeT = total_nodeT/Total_Node
                    Pro_nodeF = total_nodeF/Total_Node
                    Weighted_avr.append(true_impurity* Pro_nodeT + false_impurity* Pro_nodeF)
            best_splits_true.append(trueList)#Store both the trueList and falseList externally
            best_splits_false.append(falseList)
        Min_wieght= min(Weighted_avr)
        root_index =Weighted_avr.index(Min_wieght)
        bestAtt = attributes[root_index]
        trueList =best_splits_true[root_index]#Retrieve the trueList corresponding to bestAtt using root_index
        falseList = best_splits_false[root_index]#Retrieve the falseList corresponding to bestAtt using root_index
        attributes.remove(bestAtt)
        left = BuildTree(trueList, attributes)
        right = BuildTree(falseList , attributes)
        return Node (bestAtt,left,right)


def main():
    global insdata
    correct=0
    incorrect=0
    base_correct=0
    base_incorrect=0
    args = {'trainingF': sys.argv[1], 'testingF': sys.argv[2]}
    atridata,insdata = getdata(args['trainingF'])
    tree = BuildTree(insdata, atridata)
    tree.print_Out(" ")
    atri_testdata, ins_testdata = getdata(args['testingF'])
    for i in range(1, len( ins_testdata)):
        cl = Classify(tree,  ins_testdata[i])
        if cl== ins_testdata[i][0]:
            correct +=1
        else:
            incorrect +=1
    accuracy = float(correct)/(correct+incorrect)
    print('\n\n')
    print('Decision Tree accuracy:%s'%accuracy)
    most_class , prob = get_classes (insdata)
    for i in range(1, len(ins_testdata)):
        if most_class == ins_testdata[i][0]:
            base_correct += 1
        else:
            base_incorrect += 1
    accuracy_base=float(base_correct)/(base_correct+base_incorrect)
    print('Baseline classiﬁer accuracy: %s'%accuracy_base)

if __name__ == '__main__':
    main()
