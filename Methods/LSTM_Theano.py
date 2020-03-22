import theano.tensor as T

vectorSize=
hiddenSize = 10

def lstm(x, cm1, hm1, W):
    #  x is the input vector. cm1 is the memory state, hm1 is the hidden cell and y is the output

    hx = T.concatenate([x, hm1])
    hxSize = hx.shape[0]
    bs = 0
    Wf = W[bs: bs + hiddenSize * hxSize].reshape([hiddenSize, hxSize])
    bs += hiddenSize * hxSize
    bf = W[bs: bs + hiddenSize]
    bs += hiddenSize
    Wi = W[bs: bs + hiddenSize * hxSize].reshape([hiddenSize, hxSize])
    bs += hiddenSize * hxSize
    bi = W[bs: bs + hiddenSize]
    bs += hiddenSize
    Wc = W[bs: bs + hiddenSize * hxSize].reshape([hiddenSize, hxSize])
    bs += hiddenSize * hxSize
    bc = W[bs: bs + hiddenSize]
    bs += hiddenSize
    Wo = W[bs: bs + hiddenSize * hxSize].reshape([hiddenSize, hxSize])
    bs += hiddenSize * hxSize
    bo = W[bs: bs + hiddenSize]
    bs += hiddenSize
    Wy = W[bs: bs + vectorSize * hiddenSize].reshape([vectorSize, hiddenSize])
    bs += vectorSize * hiddenSize
    by = W[bs: bs + vectorSize]
    bs += vectorSize
    ft = T.nnet.sigmoid(Wf.dot(hx) + bf)
    it = T.nnet.sigmoid(Wi.dot(hx) + bi)
    ct = T.tanh(Wc.dot(hx) + bc)
    ot = T.nnet.sigmoid(Wo.dot(hx) + bo)
    c = ft * cm1 + it * ct
    h = ot * T.tanh(c)
    y = Wy.dot(h) + by
    return [y, c, h, y]

tResult, tUpdates = theano.scan(lstm,
                                outputs_info = [None,
                                        T.zeros(hiddenSize),
                                        T.zeros(hiddenSize),
                                        T.zeros(vectorSize)],
                                sequences = [dict(input = tx)],
                                non_sequences = [tW])