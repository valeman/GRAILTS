import numpy as np;

def reorder(arr,index): 
    temp = [0] * len(arr); 
    result = [0] * len(arr); 
  
    # arr[i] should be 
        # present at index[i] index 
    for i in range(len(arr)): 
        temp[i] = arr[index[i]] 
  
    # Copy temp[] to arr[] 
    for i in range(len(arr)): 
        result[i] = temp[i] 
    
    return result;


def OneNN2(train,trainclasses,test,testclasses,delta,epsilon = 5):
  acc = 0
  pruning_power = 0;
  for ids in range(len(test)):
    best_so_far = float('inf');
    distance_lb = np.zeros(len(train));

    lbdistcomp = 0;
    for i in range(len(train)):
      distance_lb[i] = LBKeogh(train[i],test[ids],delta);
      lbdistcomp = 1 + lbdistcomp;


    ordering = np.argsort(distance_lb);
    distance_lb.sort();
    traindata = reorder(train,ordering)
    trainclasses_r = reorder(trainclasses,ordering);
  
    actualdistcomp = 0;
    for i in range(len(train)):
      if distance_lb[i] < best_so_far:
        distance = dtwabs_scb(traindata[i], test[ids],delta);
        actualdistcomp = actualdistcomp + 1;
        if distance < best_so_far:
          dclass = trainclasses_r[i]
          best_so_far = distance;
    if (testclasses[ids] == dclass):
      acc = acc + 1;

  pruning_power = 1 - (actualdistcomp/lbdistcomp);
  acc = acc / len(test);
    
  return acc, pruning_power;

def OneNN3(train,trainclasses,test,testclasses,epsilon):
  acc = 0
  for ids in range(len(test)):
    best_so_far = float('inf');
    for i in range(len(train)):
        distance = dtw_abs(train[i], test[ids]);
        print(distance);
        if distance < best_so_far:
          dclass = trainclasses[i]
          best_so_far = distance;


    if (testclasses[ids] == dclass):
      acc = acc + 1;
  acc = acc / len(test);
  return acc