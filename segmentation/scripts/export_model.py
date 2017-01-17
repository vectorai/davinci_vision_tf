from keras import backend as K
from tensorflow_serving.session_bundle import exporter
from keras.models import model_from_config
import tensorflow as tf

def export(sess,previos_model,export_path,export_version):
    K.set_learning_phase(0)  # all new operations will be in test mode from now on

    # serialize the model and get its weights, for quick re-building
    config = previous_model.get_config()
    weights = previous_model.get_weights()

    # re-build a model where the learning phase is now hard-coded to 0
    new_model = model_from_config(config)
    new_model.set_weights(weights)

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=model.input,
                                                          scores_tensor=model.output)
    model_exporter.init(sess.graph.as_graph_def(),
                                    default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)

if __name__=="__main__":
    import sys
    sess = tf.Session()
    K.set_session(sess)

    mod=load_model(sys.argv[1])
    export(mod,sys.argv[2],sys.argv[3],sess)
