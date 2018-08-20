import pandas as pd
import numpy as np

plot_time_reward="/home/deve/workspace/RL/RL/1.csv"
train=pd.read_csv('/home/deve/workspace/RL/RL/before_plot_time_data.csv')
#train=pd.read_csv('/home/deve/workspace/RL/RL/afterplot_time_data.csv')
train=np.array(train)
input1=np.array(train[:,0]).reshape(len(train[:,0]))

input2=train[:,1]
input2=np.array(train[:,1]).reshape(len(train[:,1]))
print (train)
#线性平滑
def linearSmooth5(input, N):
    out=[0]*len(input)
    if (N < 5):
        for i in range(N):
            out[i] = input[i]
    else:
        out[0] = ( 3.0 * input[0] + 2.0 * input[1] + input[2] - input[4] ) / 5.0
        out[1] = ( 4.0 * input[0] + 3.0 * input[1] + 2 * input[2] + input[3] ) / 10.0
        for i in range(2,N):
            out[i] = ( input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2] ) / 5.0
        out[N - 2] = ( 4.0 * input[N - 1] + 3.0 * input[N - 2] + 2 * input[N - 3] + input[N - 4] ) / 10.0
        out[N - 1] = ( 3.0 * input[N - 1] + 2.0 * input[N - 2] + input[N - 3] - input[N - 5] ) / 5.0
    return out

def linearSmooth7(input, N):
    out = [0] * len(input)
    if (N < 7):
        for i in range(N):
            out[i] = input[i]
    else:

        out[0] = ( 13.0 * input[0] + 10.0 * input[1] + 7.0 * input[2] + 4.0 * input[3] +
                   input[4] - 2.0 * input[5] - 5.0 * input[6] ) / 28.0

        out[1] = ( 5.0 * input[0] + 4.0 * input[1] + 3 * input[2] + 2 * input[3] +
                   input[4] - input[6] ) / 14.0

        out[2] = ( 7.0 * input[0] + 6.0 * input [1] + 5.0 * input[2] + 4.0 * input[3] +
                  3.0 * input[4] + 2.0 * input[5] + input[6] ) / 28.0
        for i in range(3, N - 3):
            out[i] = ( input[i - 3] + input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2] + input[i + 3] ) / 7.0

        out[N - 3] = ( 7.0 * input[N - 1] + 6.0 * input [N - 2] + 5.0 * input[N - 3] +
                      4.0 * input[N - 4] + 3.0 * input[N - 5] + 2.0 * input[N - 6] + input[N - 7] ) / 28.0

        out[N - 2] = ( 5.0 * input[N - 1] + 4.0 * input[N - 2] + 3.0 * input[N - 3] +
                      2.0 * input[N - 4] + input[N - 5] - input[N - 7] ) / 14.0

        out[N - 1] = ( 13.0 * input[N - 1] + 10.0 * input[N - 2] + 7.0 * input[N - 3] +
                      4 * input[N - 4] + input[N - 5] - 2 * input[N - 6] - 5 * input[N - 7] ) / 28.0
    return out


#五点三次平滑
def cubicSmooth5(input, N):
    out = [0] * len(input)
    if (N < 5):
        for i in range(N):
            out[i] = input[i]
    else:
        out[0] = (69.0 * input[0] + 4.0 * input[1] - 6.0 * input[2] + 4.0 * input[3] - input[4]) / 70.0
        out[1] = (2.0 * input[0] + 27.0 * input[1] + 12.0 * input[2] - 8.0 * input[3] + 2.0 * input[4]) / 35.0
        for i in range(2,N-2):
            out[i] = (-3.0 * ( input[i - 2] + input[i + 2])+ 12.0 * ( input[i - 1] + input[i + 1]) + 17.0 * input[i] ) / 35.0
        out[N - 2] = (2.0 * input[N - 5] - 8.0 * input[N - 4] + 12.0 * input[N - 3] + 27.0 * input[N - 2] + 2.0 * input[N - 1]) / 35.0
        out[N - 1] = (- input[N - 5] + 4.0 * input[N - 4] - 6.0 * input[N - 3] + 4.0 * input[N - 2] + 69.0 * input[N - 1]) / 70.0
    return out






out = {"time": []}
out1=linearSmooth7(input1,len(input1))
#out2=linearSmooth5(input2,len(input2))
for i in range(len(out1)):
    out["time"].append(out1[i])
plot = pd.DataFrame(data=out)
plot.to_csv(plot_time_reward, index=False)
#a=np.mean(rewards)
#out["rewards"].append(out2)