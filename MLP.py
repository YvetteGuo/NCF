import mxnet as mx

def mlp_model(model_layers, max_user, max_item, dense):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    stype = 'default' if dense else 'row_sparse'
    sparse_grad = not dense
    user_weight = mx.sym.Variable('user_weight', stype=stype)
    item_weight = mx.sym.Variable('item_weight', stype=stype)
    
    mlp_user_latent = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=model_layers[0]//2)
    mlp_item_latent = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=model_layers[0]//2)
                            
    # Concatenation of two latent features
    mlp_vector=mx.sym.concat(mlp_user_latent, mlp_item_latent, dim=1) #横轴上的merge 如[[1,2],[3,4]] [[5,6],[7,8]]->[[1,2,5,6],[3,4,7,8]]

    num_layer = len(model_layers)  # Number of layers in the MLP, model_layers是一个参数
    for layer in range(1, num_layer):
        mlp_vector = mx.sym.FullyConnected(data=mlp_vector, num_hidden=model_layers[layer])
        mlp_vector=mx.sym.Activation(data=mlp_vector, act_type='relu')
      
    # Final prediction layer
    pred = mx.sym.FullyConnected(data=mlp_vector, num_hidden=1) 
    # pred = mx.sym.Activation(data=pred, act_type='sigmoid')
    pred = mx.sym.Flatten(data=pred)
    # loss layer
    pred = mx.sym.LogisticRegressionOutput(data=pred, label=score)
    return pred
    