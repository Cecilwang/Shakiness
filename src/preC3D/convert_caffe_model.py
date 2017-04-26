import caffe_pb2 as caffe
import numpy as np
from C3D import C3D

def reindex(x):
    # https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L90-L115
    # invert the last three axes
    if x.ndim != 5:
        print ("[Error] Input to reindex must be 5D nparray.")
        return None

    N = x.shape[0]
    C = x.shape[1]
    L = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]
    y = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for l in range(L):
                for h in range(H):
                    for w in range(W):
                        y[n, c, l, h, w] = x[n, c,
                                                   L - l - 1,
                                                   H - h - 1,
                                                   W - w - 1]
    return y

def convert_dense(w):
    # kernel: (8192, 4096): (512x1x4x4, 4096) -> (1x4x4x512, 4096)
    wo = np.zeros_like(w)
    for i in range(w.shape[1]):
        wi = np.squeeze(w[:,i])
        wo[:,i] = np.transpose(np.reshape(wi, (512,4,4)), (1, 2, 0)).flatten()
    return wo

def main():

    dim_ordering = 'tf'
    print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
            dim_ordering))

    # get C3D model placeholder
    model = C3D().create([16, 112, 112, 3])
    model.summary()
    # input caffe model
    caffe_model_filename = 'conv3d_deepnetA_sport1m_iter_1900000'


    # read caffe model
    print ("-" * 19)
    print ("Reading model file={}...".format(caffe_model_filename))
    p = caffe.NetParameter()
    p.ParseFromString(open(caffe_model_filename, 'rb').read())

    params = []
    print ("-" * 19)
    print ("Converting model...")

    # read every conv/fc layer and append to "params" list
    for i in range(len(p.layers)):
        layer = p.layers[i]
        # skip non-conv/fc layers
        if 'conv' not in layer.name and 'fc' not in layer.name:
            continue
        print ("[Info] Massaging \"{}\" layer...".format(layer.name))
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num,
            layer.blobs[0].channels,
            layer.blobs[0].length,
            layer.blobs[0].height,
            layer.blobs[0].width,
            )
        if 'conv' in layer.name:
            weights_p = np.transpose(weights_p, (2, 3, 4, 1, 0))
        elif 'fc' in layer.name:
            weights_p = weights_p[0, 0, 0, :, :].T
            if 'fc6' in layer.name:
                print("[Info] First FC layer after flattening layer needs "
                      "special care...")
                weights_p = convert_dense(weights_p)
        params.append([weights_p, weights_b])

    valid_layer_count = 0
    for layer_indx in range(len(model.layers)):
        layer_name = model.layers[layer_indx].name
        if 'conv' in layer_name or 'fc' in layer_name:
            print ("[Info] Transplanting \"{}\" layer...".format(layer_name))
            model.layers[layer_indx].set_weights(params[valid_layer_count])
            valid_layer_count += 1

    print ("-" * 19)
    print ("Saving pre-trained model weights as preC3D.hdf5.")
    model.save('preC3D-c.hdf5', overwrite=True)
    print ("-" * 39)
    print ("Conversion done!")
    print ("-" * 39)

if __name__ == '__main__':
    main()
