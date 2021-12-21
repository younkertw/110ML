import tensorflow as tf
import tensorflow.compat.v1 as tf #Eric
tf.disable_v2_behavior() #Eric

def bi_lstm(Input_X, num_hidden, MAXLEN, keep_prob):
	

	x = tf.unstack(Input_X, MAXLEN, 1)
	
	lstm_fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_hidden//2, forget_bias=1.0) #Eric
	lstm_fw_cell= tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,input_keep_prob=keep_prob, output_keep_prob=keep_prob) #Eric
	
	lstm_bw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_hidden//2, forget_bias=1.0) #Eric
	lstm_bw_cell= tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,input_keep_prob=keep_prob, output_keep_prob=keep_prob) #Eric
	
	outputs, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
												  dtype=tf.float32,
												  )
	#outputs, _ = tf.compat.v1.nn.static_rnn(lstm_fw_cell, x, dtype=tf.float32)	 #Eric
	outputs = tf.compat.v1.stack(outputs,axis = 1)  #Eric

	return outputs

