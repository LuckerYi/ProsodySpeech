class Weightlist(object):
	"""docstring for Weightlist"""
	def __init__(self):
		super(Weightlist, self).__init__()
		self.MIfrozen_list=[
		'speaker_embedding.weight',
		'spkemb_projection.weight',
		'spkemb_projection.bias',
		'encoder.embed.0.weight',
		'encoder.embed.1.alpha',
		'encoder.encoders.0.self_attn.linear_q.weight',
		'encoder.encoders.0.self_attn.linear_q.bias',
		'encoder.encoders.0.self_attn.linear_k.weight',
		'encoder.encoders.0.self_attn.linear_k.bias',
		'encoder.encoders.0.self_attn.linear_v.weight',
		'encoder.encoders.0.self_attn.linear_v.bias',
		'encoder.encoders.0.self_attn.linear_out.weight',
		'encoder.encoders.0.self_attn.linear_out.bias',
		'encoder.encoders.0.feed_forward.w_1.weight',
		'encoder.encoders.0.feed_forward.w_1.bias',
		'encoder.encoders.0.feed_forward.w_2.weight',
		'encoder.encoders.0.feed_forward.w_2.bias',
		'encoder.encoders.0.norm1.w.weight',
		'encoder.encoders.0.norm1.w.bias',
		'encoder.encoders.0.norm1.b.weight',
		'encoder.encoders.0.norm1.b.bias',
		'encoder.encoders.0.norm2.w.weight',
		'encoder.encoders.0.norm2.w.bias',
		'encoder.encoders.0.norm2.b.weight',
		'encoder.encoders.0.norm2.b.bias',
		'encoder.encoders.1.self_attn.linear_q.weight',
		'encoder.encoders.1.self_attn.linear_q.bias',
		'encoder.encoders.1.self_attn.linear_k.weight',
		'encoder.encoders.1.self_attn.linear_k.bias',
		'encoder.encoders.1.self_attn.linear_v.weight',
		'encoder.encoders.1.self_attn.linear_v.bias',
		'encoder.encoders.1.self_attn.linear_out.weight',
		'encoder.encoders.1.self_attn.linear_out.bias',
		'encoder.encoders.1.feed_forward.w_1.weight',
		'encoder.encoders.1.feed_forward.w_1.bias',
		'encoder.encoders.1.feed_forward.w_2.weight',
		'encoder.encoders.1.feed_forward.w_2.bias',
		'encoder.encoders.1.norm1.w.weight',
		'encoder.encoders.1.norm1.w.bias',
		'encoder.encoders.1.norm1.b.weight',
		'encoder.encoders.1.norm1.b.bias',
		'encoder.encoders.1.norm2.w.weight',
		'encoder.encoders.1.norm2.w.bias',
		'encoder.encoders.1.norm2.b.weight',
		'encoder.encoders.1.norm2.b.bias',
		'encoder.encoders.2.self_attn.linear_q.weight',
		'encoder.encoders.2.self_attn.linear_q.bias',
		'encoder.encoders.2.self_attn.linear_k.weight',
		'encoder.encoders.2.self_attn.linear_k.bias',
		'encoder.encoders.2.self_attn.linear_v.weight',
		'encoder.encoders.2.self_attn.linear_v.bias',
		'encoder.encoders.2.self_attn.linear_out.weight',
		'encoder.encoders.2.self_attn.linear_out.bias',
		'encoder.encoders.2.feed_forward.w_1.weight',
		'encoder.encoders.2.feed_forward.w_1.bias',
		'encoder.encoders.2.feed_forward.w_2.weight',
		'encoder.encoders.2.feed_forward.w_2.bias',
		'encoder.encoders.2.norm1.w.weight',
		'encoder.encoders.2.norm1.w.bias',
		'encoder.encoders.2.norm1.b.weight',
		'encoder.encoders.2.norm1.b.bias',
		'encoder.encoders.2.norm2.w.weight',
		'encoder.encoders.2.norm2.w.bias',
		'encoder.encoders.2.norm2.b.weight',
		'encoder.encoders.2.norm2.b.bias',
		'encoder.encoders.3.self_attn.linear_q.weight',
		'encoder.encoders.3.self_attn.linear_q.bias',
		'encoder.encoders.3.self_attn.linear_k.weight',
		'encoder.encoders.3.self_attn.linear_k.bias',
		'encoder.encoders.3.self_attn.linear_v.weight',
		'encoder.encoders.3.self_attn.linear_v.bias',
		'encoder.encoders.3.self_attn.linear_out.weight',
		'encoder.encoders.3.self_attn.linear_out.bias',
		'encoder.encoders.3.feed_forward.w_1.weight',
		'encoder.encoders.3.feed_forward.w_1.bias',
		'encoder.encoders.3.feed_forward.w_2.weight',
		'encoder.encoders.3.feed_forward.w_2.bias',
		'encoder.encoders.3.norm1.w.weight',
		'encoder.encoders.3.norm1.w.bias',
		'encoder.encoders.3.norm1.b.weight',
		'encoder.encoders.3.norm1.b.bias',
		'encoder.encoders.3.norm2.w.weight',
		'encoder.encoders.3.norm2.w.bias',
		'encoder.encoders.3.norm2.b.weight',
		'encoder.encoders.3.norm2.b.bias',
		'encoder.encoders.4.self_attn.linear_q.weight',
		'encoder.encoders.4.self_attn.linear_q.bias',
		'encoder.encoders.4.self_attn.linear_k.weight',
		'encoder.encoders.4.self_attn.linear_k.bias',
		'encoder.encoders.4.self_attn.linear_v.weight',
		'encoder.encoders.4.self_attn.linear_v.bias',
		'encoder.encoders.4.self_attn.linear_out.weight',
		'encoder.encoders.4.self_attn.linear_out.bias',
		'encoder.encoders.4.feed_forward.w_1.weight',
		'encoder.encoders.4.feed_forward.w_1.bias',
		'encoder.encoders.4.feed_forward.w_2.weight',
		'encoder.encoders.4.feed_forward.w_2.bias',
		'encoder.encoders.4.norm1.w.weight',
		'encoder.encoders.4.norm1.w.bias',
		'encoder.encoders.4.norm1.b.weight',
		'encoder.encoders.4.norm1.b.bias',
		'encoder.encoders.4.norm2.w.weight',
		'encoder.encoders.4.norm2.w.bias',
		'encoder.encoders.4.norm2.b.weight',
		'encoder.encoders.4.norm2.b.bias',
		'encoder.encoders.5.self_attn.linear_q.weight',
		'encoder.encoders.5.self_attn.linear_q.bias',
		'encoder.encoders.5.self_attn.linear_k.weight',
		'encoder.encoders.5.self_attn.linear_k.bias',
		'encoder.encoders.5.self_attn.linear_v.weight',
		'encoder.encoders.5.self_attn.linear_v.bias',
		'encoder.encoders.5.self_attn.linear_out.weight',
		'encoder.encoders.5.self_attn.linear_out.bias',
		'encoder.encoders.5.feed_forward.w_1.weight',
		'encoder.encoders.5.feed_forward.w_1.bias',
		'encoder.encoders.5.feed_forward.w_2.weight',
		'encoder.encoders.5.feed_forward.w_2.bias',
		'encoder.encoders.5.norm1.w.weight',
		'encoder.encoders.5.norm1.w.bias',
		'encoder.encoders.5.norm1.b.weight',
		'encoder.encoders.5.norm1.b.bias',
		'encoder.encoders.5.norm2.w.weight',
		'encoder.encoders.5.norm2.w.bias',
		'encoder.encoders.5.norm2.b.weight',
		'encoder.encoders.5.norm2.b.bias',
		'encoder.after_norm.w.weight',
		'encoder.after_norm.w.bias',
		'encoder.after_norm.b.weight',
		'encoder.after_norm.b.bias',
		'projection.weight',
		'projection.bias',
		]
		
		self.refine_list=[
        'speaker_embedding.weight',#embedding LUT
        'spkemb_projection.weight',#uesd for embedding to linear projecttion
        'spkemb_projection.bias',#uesd for embedding to linear projecttion
        'projection.weight',#uesd for concat or add to linear projecttion
        'projection.bias',#uesd for concat or add to linear projecttion
        'encoder.encoders.0.norm1.w.weight',
        'encoder.encoders.0.norm1.w.bias',
        'encoder.encoders.0.norm1.b.weight',
        'encoder.encoders.0.norm1.b.bias',
        'encoder.encoders.0.norm2.w.weight',
        'encoder.encoders.0.norm2.w.bias',
        'encoder.encoders.0.norm2.b.weight',
        'encoder.encoders.0.norm2.b.bias',
        'encoder.encoders.1.norm1.w.weight',
        'encoder.encoders.1.norm1.w.bias',
        'encoder.encoders.1.norm1.b.weight',
        'encoder.encoders.1.norm1.b.bias',
        'encoder.encoders.1.norm2.w.weight',
        'encoder.encoders.1.norm2.w.bias',
        'encoder.encoders.1.norm2.b.weight',
        'encoder.encoders.1.norm2.b.bias',
        'encoder.encoders.2.norm1.w.weight',
        'encoder.encoders.2.norm1.w.bias',
        'encoder.encoders.2.norm1.b.weight',
        'encoder.encoders.2.norm1.b.bias',
        'encoder.encoders.2.norm2.w.weight',
        'encoder.encoders.2.norm2.w.bias',
        'encoder.encoders.2.norm2.b.weight',
        'encoder.encoders.2.norm2.b.bias',
        'encoder.encoders.3.norm1.w.weight',
        'encoder.encoders.3.norm1.w.bias',
        'encoder.encoders.3.norm1.b.weight',
        'encoder.encoders.3.norm1.b.bias',
        'encoder.encoders.3.norm2.w.weight',
        'encoder.encoders.3.norm2.w.bias',
        'encoder.encoders.3.norm2.b.weight',
        'encoder.encoders.3.norm2.b.bias',
        'encoder.encoders.4.norm1.w.weight',
        'encoder.encoders.4.norm1.w.bias',
        'encoder.encoders.4.norm1.b.weight',
        'encoder.encoders.4.norm1.b.bias',
        'encoder.encoders.4.norm2.w.weight',
        'encoder.encoders.4.norm2.w.bias',
        'encoder.encoders.4.norm2.b.weight',
        'encoder.encoders.4.norm2.b.bias',
        'encoder.encoders.5.norm1.w.weight',
        'encoder.encoders.5.norm1.w.bias',
        'encoder.encoders.5.norm1.b.weight',
        'encoder.encoders.5.norm1.b.bias',
        'encoder.encoders.5.norm2.w.weight',
        'encoder.encoders.5.norm2.w.bias',
        'encoder.encoders.5.norm2.b.weight',
        'encoder.encoders.5.norm2.b.bias',
        'encoder.after_norm.w.weight',
        'encoder.after_norm.w.bias',
        'encoder.after_norm.b.weight',
        'encoder.after_norm.b.bias',
        'duration_predictor.norm.0.w.weight',
        'duration_predictor.norm.0.w.bias',
        'duration_predictor.norm.0.b.weight',
        'duration_predictor.norm.0.b.bias',
        'duration_predictor.norm.1.w.weight',
        'duration_predictor.norm.1.w.bias',
        'duration_predictor.norm.1.b.weight',
        'duration_predictor.norm.1.b.bias',
        'decoder.encoders.0.norm1.w.weight',
        'decoder.encoders.0.norm1.w.bias',
        'decoder.encoders.0.norm1.b.weight',
        'decoder.encoders.0.norm1.b.bias',
        'decoder.encoders.0.norm2.w.weight',
        'decoder.encoders.0.norm2.w.bias',
        'decoder.encoders.0.norm2.b.weight',
        'decoder.encoders.0.norm2.b.bias',
        'decoder.encoders.1.norm1.w.weight',
        'decoder.encoders.1.norm1.w.bias',
        'decoder.encoders.1.norm1.b.weight',
        'decoder.encoders.1.norm1.b.bias',
        'decoder.encoders.1.norm2.w.weight',
        'decoder.encoders.1.norm2.w.bias',
        'decoder.encoders.1.norm2.b.weight',
        'decoder.encoders.1.norm2.b.bias',
        'decoder.encoders.2.norm1.w.weight',
        'decoder.encoders.2.norm1.w.bias',
        'decoder.encoders.2.norm1.b.weight',
        'decoder.encoders.2.norm1.b.bias',
        'decoder.encoders.2.norm2.w.weight',
        'decoder.encoders.2.norm2.w.bias',
        'decoder.encoders.2.norm2.b.weight',
        'decoder.encoders.2.norm2.b.bias',
        'decoder.encoders.3.norm1.w.weight',
        'decoder.encoders.3.norm1.w.bias',
        'decoder.encoders.3.norm1.b.weight',
        'decoder.encoders.3.norm1.b.bias',
        'decoder.encoders.3.norm2.w.weight',
        'decoder.encoders.3.norm2.w.bias',
        'decoder.encoders.3.norm2.b.weight',
        'decoder.encoders.3.norm2.b.bias',
        'decoder.encoders.4.norm1.w.weight',
        'decoder.encoders.4.norm1.w.bias',
        'decoder.encoders.4.norm1.b.weight',
        'decoder.encoders.4.norm1.b.bias',
        'decoder.encoders.4.norm2.w.weight',
        'decoder.encoders.4.norm2.w.bias',
        'decoder.encoders.4.norm2.b.weight',
        'decoder.encoders.4.norm2.b.bias',
        'decoder.encoders.5.norm1.w.weight',
        'decoder.encoders.5.norm1.w.bias',
        'decoder.encoders.5.norm1.b.weight',
        'decoder.encoders.5.norm1.b.bias',
        'decoder.encoders.5.norm2.w.weight',
        'decoder.encoders.5.norm2.w.bias',
        'decoder.encoders.5.norm2.b.weight',
        'decoder.encoders.5.norm2.b.bias',
        'decoder.after_norm.w.weight',
        'decoder.after_norm.w.bias',
        'decoder.after_norm.b.weight',
        'decoder.after_norm.b.bias',
		]
		
		self.refine_list_gaussian=[
		'sigma_predictor.norm.0.w.weight',
		'sigma_predictor.norm.0.w.bias',
		'sigma_predictor.norm.0.b.weight',
		'sigma_predictor.norm.0.b.bias',
		'sigma_predictor.norm.1.w.weight',
		'sigma_predictor.norm.1.w.bias',
		'sigma_predictor.norm.1.b.weight',
		'sigma_predictor.norm.1.b.bias',
		]

		self.choosestl_list=[
		'gst.choosestl.choose_mha.linear_q.weight',
		'gst.choosestl.choose_mha.linear_q.bias',
		'gst.choosestl.choose_mha.linear_k.weight',
		'gst.choosestl.choose_mha.linear_k.bias',
		'gst.choosestl.choose_mha.linear_v.weight',
		'gst.choosestl.choose_mha.linear_v.bias',
		'gst.choosestl.choose_mha.linear_out.weight',
		'gst.choosestl.choose_mha.linear_out.bias',
		]

		self.gst_reference_encoder_list =[
		'gst.ref_enc.style_encoders.0.self_attn.linear_q.weight',
		'gst.ref_enc.style_encoders.0.self_attn.linear_q.bias',
		'gst.ref_enc.style_encoders.0.self_attn.linear_k.weight',
		'gst.ref_enc.style_encoders.0.self_attn.linear_k.bias',
		'gst.ref_enc.style_encoders.0.self_attn.linear_v.weight',
		'gst.ref_enc.style_encoders.0.self_attn.linear_v.bias',
		'gst.ref_enc.style_encoders.0.self_attn.linear_out.weight',
		'gst.ref_enc.style_encoders.0.self_attn.linear_out.bias',
		'gst.ref_enc.style_encoders.0.feed_forward.w_1.weight',
		'gst.ref_enc.style_encoders.0.feed_forward.w_1.bias',
		'gst.ref_enc.style_encoders.0.feed_forward.w_2.weight',
		'gst.ref_enc.style_encoders.0.feed_forward.w_2.bias',
		'gst.ref_enc.style_encoders.0.norm1.weight',
		'gst.ref_enc.style_encoders.0.norm1.bias',
		'gst.ref_enc.style_encoders.0.norm2.weight',
		'gst.ref_enc.style_encoders.0.norm2.bias',
		'gst.ref_enc.style_encoders.1.self_attn.linear_q.weight',
		'gst.ref_enc.style_encoders.1.self_attn.linear_q.bias',
		'gst.ref_enc.style_encoders.1.self_attn.linear_k.weight',
		'gst.ref_enc.style_encoders.1.self_attn.linear_k.bias',
		'gst.ref_enc.style_encoders.1.self_attn.linear_v.weight',
		'gst.ref_enc.style_encoders.1.self_attn.linear_v.bias',
		'gst.ref_enc.style_encoders.1.self_attn.linear_out.weight',
		'gst.ref_enc.style_encoders.1.self_attn.linear_out.bias',
		'gst.ref_enc.style_encoders.1.feed_forward.w_1.weight',
		'gst.ref_enc.style_encoders.1.feed_forward.w_1.bias',
		'gst.ref_enc.style_encoders.1.feed_forward.w_2.weight',
		'gst.ref_enc.style_encoders.1.feed_forward.w_2.bias',
		'gst.ref_enc.style_encoders.1.norm1.weight',
		'gst.ref_enc.style_encoders.1.norm1.bias',
		'gst.ref_enc.style_encoders.1.norm2.weight',
		'gst.ref_enc.style_encoders.1.norm2.bias',
		'gst.ref_enc.style_encoders.2.self_attn.linear_q.weight',
		'gst.ref_enc.style_encoders.2.self_attn.linear_q.bias',
		'gst.ref_enc.style_encoders.2.self_attn.linear_k.weight',
		'gst.ref_enc.style_encoders.2.self_attn.linear_k.bias',
		'gst.ref_enc.style_encoders.2.self_attn.linear_v.weight',
		'gst.ref_enc.style_encoders.2.self_attn.linear_v.bias',
		'gst.ref_enc.style_encoders.2.self_attn.linear_out.weight',
		'gst.ref_enc.style_encoders.2.self_attn.linear_out.bias',
		'gst.ref_enc.style_encoders.2.feed_forward.w_1.weight',
		'gst.ref_enc.style_encoders.2.feed_forward.w_1.bias',
		'gst.ref_enc.style_encoders.2.feed_forward.w_2.weight',
		'gst.ref_enc.style_encoders.2.feed_forward.w_2.bias',
		'gst.ref_enc.style_encoders.2.norm1.weight',
		'gst.ref_enc.style_encoders.2.norm1.bias',
		'gst.ref_enc.style_encoders.2.norm2.weight',
		'gst.ref_enc.style_encoders.2.norm2.bias',
		'gst.ref_enc.style_encoders.3.self_attn.linear_q.weight',
		'gst.ref_enc.style_encoders.3.self_attn.linear_q.bias',
		'gst.ref_enc.style_encoders.3.self_attn.linear_k.weight',
		'gst.ref_enc.style_encoders.3.self_attn.linear_k.bias',
		'gst.ref_enc.style_encoders.3.self_attn.linear_v.weight',
		'gst.ref_enc.style_encoders.3.self_attn.linear_v.bias',
		'gst.ref_enc.style_encoders.3.self_attn.linear_out.weight',
		'gst.ref_enc.style_encoders.3.self_attn.linear_out.bias',
		'gst.ref_enc.style_encoders.3.feed_forward.w_1.weight',
		'gst.ref_enc.style_encoders.3.feed_forward.w_1.bias',
		'gst.ref_enc.style_encoders.3.feed_forward.w_2.weight',
		'gst.ref_enc.style_encoders.3.feed_forward.w_2.bias',
		'gst.ref_enc.style_encoders.3.norm1.weight',
		'gst.ref_enc.style_encoders.3.norm1.bias',
		'gst.ref_enc.style_encoders.3.norm2.weight',
		'gst.ref_enc.style_encoders.3.norm2.bias',
		'gst.ref_enc.mha_linear.weight',
		'gst.ref_enc.mha_linear.bias',
		'gst.stl.gst_embs',
		'gst.stl.mha.linear_q.weight',
		'gst.stl.mha.linear_q.bias',
		'gst.stl.mha.linear_k.weight',
		'gst.stl.mha.linear_k.bias',
		'gst.stl.mha.linear_v.weight',
		'gst.stl.mha.linear_v.bias',
		'gst.stl.mha.linear_out.weight',
		'gst.stl.mha.linear_out.bias',
		]

		self.refine_list_f0=[
		'pitch_predictor.norm.0.w.weight',
		'pitch_predictor.norm.0.w.bias',
		'pitch_predictor.norm.0.b.weight',
		'pitch_predictor.norm.0.b.bias',
		'pitch_predictor.norm.1.w.weight',
		'pitch_predictor.norm.1.w.bias',
		'pitch_predictor.norm.1.b.weight',
		'pitch_predictor.norm.1.b.bias',
		]

		self.conv_gst_reference_encoder_list =[
		'gst.ref_enc.convs.0.weight',
		'gst.ref_enc.convs.0.bias',
		'gst.ref_enc.convs.2.weight',
		'gst.ref_enc.convs.2.bias',
		'gst.ref_enc.convs.4.weight',
		'gst.ref_enc.convs.4.bias',
		'gst.ref_enc.convs.6.weight',
		'gst.ref_enc.convs.6.bias',
		'gst.ref_enc.convs.8.weight',
		'gst.ref_enc.convs.8.bias',
		'gst.ref_enc.convs.10.weight',
		'gst.ref_enc.convs.10.bias',
		'gst.ref_enc.gru.weight_ih_l0',
		'gst.ref_enc.gru.weight_hh_l0',
		'gst.ref_enc.gru.bias_ih_l0',
		'gst.ref_enc.gru.bias_hh_l0',
		'gst.stl.gst_embs',
		'gst.stl.mha.linear_q.weight',
		'gst.stl.mha.linear_q.bias',
		'gst.stl.mha.linear_k.weight',
		'gst.stl.mha.linear_k.bias',
		'gst.stl.mha.linear_v.weight',
		'gst.stl.mha.linear_v.bias',
		'gst.stl.mha.linear_out.weight',
		'gst.stl.mha.linear_out.bias',
		]

		self.full_weight_list =[
		'speaker_embedding.weight',
		'spkemb_projection.weight',
		'spkemb_projection.bias',
		'encoder.embed.0.weight',
		'encoder.embed.1.alpha',
		'encoder.encoders.0.self_attn.linear_q.weight',
		'encoder.encoders.0.self_attn.linear_q.bias',
		'encoder.encoders.0.self_attn.linear_k.weight',
		'encoder.encoders.0.self_attn.linear_k.bias',
		'encoder.encoders.0.self_attn.linear_v.weight',
		'encoder.encoders.0.self_attn.linear_v.bias',
		'encoder.encoders.0.self_attn.linear_out.weight',
		'encoder.encoders.0.self_attn.linear_out.bias',
		'encoder.encoders.0.feed_forward.w_1.weight',
		'encoder.encoders.0.feed_forward.w_1.bias',
		'encoder.encoders.0.feed_forward.w_2.weight',
		'encoder.encoders.0.feed_forward.w_2.bias',
		'encoder.encoders.0.norm1.w.weight',
		'encoder.encoders.0.norm1.w.bias',
		'encoder.encoders.0.norm1.b.weight',
		'encoder.encoders.0.norm1.b.bias',
		'encoder.encoders.0.norm2.w.weight',
		'encoder.encoders.0.norm2.w.bias',
		'encoder.encoders.0.norm2.b.weight',
		'encoder.encoders.0.norm2.b.bias',
		'encoder.encoders.1.self_attn.linear_q.weight',
		'encoder.encoders.1.self_attn.linear_q.bias',
		'encoder.encoders.1.self_attn.linear_k.weight',
		'encoder.encoders.1.self_attn.linear_k.bias',
		'encoder.encoders.1.self_attn.linear_v.weight',
		'encoder.encoders.1.self_attn.linear_v.bias',
		'encoder.encoders.1.self_attn.linear_out.weight',
		'encoder.encoders.1.self_attn.linear_out.bias',
		'encoder.encoders.1.feed_forward.w_1.weight',
		'encoder.encoders.1.feed_forward.w_1.bias',
		'encoder.encoders.1.feed_forward.w_2.weight',
		'encoder.encoders.1.feed_forward.w_2.bias',
		'encoder.encoders.1.norm1.w.weight',
		'encoder.encoders.1.norm1.w.bias',
		'encoder.encoders.1.norm1.b.weight',
		'encoder.encoders.1.norm1.b.bias',
		'encoder.encoders.1.norm2.w.weight',
		'encoder.encoders.1.norm2.w.bias',
		'encoder.encoders.1.norm2.b.weight',
		'encoder.encoders.1.norm2.b.bias',
		'encoder.encoders.2.self_attn.linear_q.weight',
		'encoder.encoders.2.self_attn.linear_q.bias',
		'encoder.encoders.2.self_attn.linear_k.weight',
		'encoder.encoders.2.self_attn.linear_k.bias',
		'encoder.encoders.2.self_attn.linear_v.weight',
		'encoder.encoders.2.self_attn.linear_v.bias',
		'encoder.encoders.2.self_attn.linear_out.weight',
		'encoder.encoders.2.self_attn.linear_out.bias',
		'encoder.encoders.2.feed_forward.w_1.weight',
		'encoder.encoders.2.feed_forward.w_1.bias',
		'encoder.encoders.2.feed_forward.w_2.weight',
		'encoder.encoders.2.feed_forward.w_2.bias',
		'encoder.encoders.2.norm1.w.weight',
		'encoder.encoders.2.norm1.w.bias',
		'encoder.encoders.2.norm1.b.weight',
		'encoder.encoders.2.norm1.b.bias',
		'encoder.encoders.2.norm2.w.weight',
		'encoder.encoders.2.norm2.w.bias',
		'encoder.encoders.2.norm2.b.weight',
		'encoder.encoders.2.norm2.b.bias',
		'encoder.encoders.3.self_attn.linear_q.weight',
		'encoder.encoders.3.self_attn.linear_q.bias',
		'encoder.encoders.3.self_attn.linear_k.weight',
		'encoder.encoders.3.self_attn.linear_k.bias',
		'encoder.encoders.3.self_attn.linear_v.weight',
		'encoder.encoders.3.self_attn.linear_v.bias',
		'encoder.encoders.3.self_attn.linear_out.weight',
		'encoder.encoders.3.self_attn.linear_out.bias',
		'encoder.encoders.3.feed_forward.w_1.weight',
		'encoder.encoders.3.feed_forward.w_1.bias',
		'encoder.encoders.3.feed_forward.w_2.weight',
		'encoder.encoders.3.feed_forward.w_2.bias',
		'encoder.encoders.3.norm1.w.weight',
		'encoder.encoders.3.norm1.w.bias',
		'encoder.encoders.3.norm1.b.weight',
		'encoder.encoders.3.norm1.b.bias',
		'encoder.encoders.3.norm2.w.weight',
		'encoder.encoders.3.norm2.w.bias',
		'encoder.encoders.3.norm2.b.weight',
		'encoder.encoders.3.norm2.b.bias',
		'encoder.encoders.4.self_attn.linear_q.weight',
		'encoder.encoders.4.self_attn.linear_q.bias',
		'encoder.encoders.4.self_attn.linear_k.weight',
		'encoder.encoders.4.self_attn.linear_k.bias',
		'encoder.encoders.4.self_attn.linear_v.weight',
		'encoder.encoders.4.self_attn.linear_v.bias',
		'encoder.encoders.4.self_attn.linear_out.weight',
		'encoder.encoders.4.self_attn.linear_out.bias',
		'encoder.encoders.4.feed_forward.w_1.weight',
		'encoder.encoders.4.feed_forward.w_1.bias',
		'encoder.encoders.4.feed_forward.w_2.weight',
		'encoder.encoders.4.feed_forward.w_2.bias',
		'encoder.encoders.4.norm1.w.weight',
		'encoder.encoders.4.norm1.w.bias',
		'encoder.encoders.4.norm1.b.weight',
		'encoder.encoders.4.norm1.b.bias',
		'encoder.encoders.4.norm2.w.weight',
		'encoder.encoders.4.norm2.w.bias',
		'encoder.encoders.4.norm2.b.weight',
		'encoder.encoders.4.norm2.b.bias',
		'encoder.encoders.5.self_attn.linear_q.weight',
		'encoder.encoders.5.self_attn.linear_q.bias',
		'encoder.encoders.5.self_attn.linear_k.weight',
		'encoder.encoders.5.self_attn.linear_k.bias',
		'encoder.encoders.5.self_attn.linear_v.weight',
		'encoder.encoders.5.self_attn.linear_v.bias',
		'encoder.encoders.5.self_attn.linear_out.weight',
		'encoder.encoders.5.self_attn.linear_out.bias',
		'encoder.encoders.5.feed_forward.w_1.weight',
		'encoder.encoders.5.feed_forward.w_1.bias',
		'encoder.encoders.5.feed_forward.w_2.weight',
		'encoder.encoders.5.feed_forward.w_2.bias',
		'encoder.encoders.5.norm1.w.weight',
		'encoder.encoders.5.norm1.w.bias',
		'encoder.encoders.5.norm1.b.weight',
		'encoder.encoders.5.norm1.b.bias',
		'encoder.encoders.5.norm2.w.weight',
		'encoder.encoders.5.norm2.w.bias',
		'encoder.encoders.5.norm2.b.weight',
		'encoder.encoders.5.norm2.b.bias',
		'encoder.after_norm.w.weight',
		'encoder.after_norm.w.bias',
		'encoder.after_norm.b.weight',
		'encoder.after_norm.b.bias',
		'projection.weight',
		'projection.bias',
		'duration_predictor.conv.0.0.weight',
		'duration_predictor.conv.0.0.bias',
		'duration_predictor.conv.1.0.weight',
		'duration_predictor.conv.1.0.bias',
		'duration_predictor.norm.0.w.weight',
		'duration_predictor.norm.0.w.bias',
		'duration_predictor.norm.0.b.weight',
		'duration_predictor.norm.0.b.bias',
		'duration_predictor.norm.1.w.weight',
		'duration_predictor.norm.1.w.bias',
		'duration_predictor.norm.1.b.weight',
		'duration_predictor.norm.1.b.bias',
		'duration_predictor.linear.weight',
		'duration_predictor.linear.bias',
		'decoder.embed.0.alpha',
		'decoder.encoders.0.self_attn.linear_q.weight',
		'decoder.encoders.0.self_attn.linear_q.bias',
		'decoder.encoders.0.self_attn.linear_k.weight',
		'decoder.encoders.0.self_attn.linear_k.bias',
		'decoder.encoders.0.self_attn.linear_v.weight',
		'decoder.encoders.0.self_attn.linear_v.bias',
		'decoder.encoders.0.self_attn.linear_out.weight',
		'decoder.encoders.0.self_attn.linear_out.bias',
		'decoder.encoders.0.feed_forward.w_1.weight',
		'decoder.encoders.0.feed_forward.w_1.bias',
		'decoder.encoders.0.feed_forward.w_2.weight',
		'decoder.encoders.0.feed_forward.w_2.bias',
		'decoder.encoders.0.norm1.w.weight',
		'decoder.encoders.0.norm1.w.bias',
		'decoder.encoders.0.norm1.b.weight',
		'decoder.encoders.0.norm1.b.bias',
		'decoder.encoders.0.norm2.w.weight',
		'decoder.encoders.0.norm2.w.bias',
		'decoder.encoders.0.norm2.b.weight',
		'decoder.encoders.0.norm2.b.bias',
		'decoder.encoders.1.self_attn.linear_q.weight',
		'decoder.encoders.1.self_attn.linear_q.bias',
		'decoder.encoders.1.self_attn.linear_k.weight',
		'decoder.encoders.1.self_attn.linear_k.bias',
		'decoder.encoders.1.self_attn.linear_v.weight',
		'decoder.encoders.1.self_attn.linear_v.bias',
		'decoder.encoders.1.self_attn.linear_out.weight',
		'decoder.encoders.1.self_attn.linear_out.bias',
		'decoder.encoders.1.feed_forward.w_1.weight',
		'decoder.encoders.1.feed_forward.w_1.bias',
		'decoder.encoders.1.feed_forward.w_2.weight',
		'decoder.encoders.1.feed_forward.w_2.bias',
		'decoder.encoders.1.norm1.w.weight',
		'decoder.encoders.1.norm1.w.bias',
		'decoder.encoders.1.norm1.b.weight',
		'decoder.encoders.1.norm1.b.bias',
		'decoder.encoders.1.norm2.w.weight',
		'decoder.encoders.1.norm2.w.bias',
		'decoder.encoders.1.norm2.b.weight',
		'decoder.encoders.1.norm2.b.bias',
		'decoder.encoders.2.self_attn.linear_q.weight',
		'decoder.encoders.2.self_attn.linear_q.bias',
		'decoder.encoders.2.self_attn.linear_k.weight',
		'decoder.encoders.2.self_attn.linear_k.bias',
		'decoder.encoders.2.self_attn.linear_v.weight',
		'decoder.encoders.2.self_attn.linear_v.bias',
		'decoder.encoders.2.self_attn.linear_out.weight',
		'decoder.encoders.2.self_attn.linear_out.bias',
		'decoder.encoders.2.feed_forward.w_1.weight',
		'decoder.encoders.2.feed_forward.w_1.bias',
		'decoder.encoders.2.feed_forward.w_2.weight',
		'decoder.encoders.2.feed_forward.w_2.bias',
		'decoder.encoders.2.norm1.w.weight',
		'decoder.encoders.2.norm1.w.bias',
		'decoder.encoders.2.norm1.b.weight',
		'decoder.encoders.2.norm1.b.bias',
		'decoder.encoders.2.norm2.w.weight',
		'decoder.encoders.2.norm2.w.bias',
		'decoder.encoders.2.norm2.b.weight',
		'decoder.encoders.2.norm2.b.bias',
		'decoder.encoders.3.self_attn.linear_q.weight',
		'decoder.encoders.3.self_attn.linear_q.bias',
		'decoder.encoders.3.self_attn.linear_k.weight',
		'decoder.encoders.3.self_attn.linear_k.bias',
		'decoder.encoders.3.self_attn.linear_v.weight',
		'decoder.encoders.3.self_attn.linear_v.bias',
		'decoder.encoders.3.self_attn.linear_out.weight',
		'decoder.encoders.3.self_attn.linear_out.bias',
		'decoder.encoders.3.feed_forward.w_1.weight',
		'decoder.encoders.3.feed_forward.w_1.bias',
		'decoder.encoders.3.feed_forward.w_2.weight',
		'decoder.encoders.3.feed_forward.w_2.bias',
		'decoder.encoders.3.norm1.w.weight',
		'decoder.encoders.3.norm1.w.bias',
		'decoder.encoders.3.norm1.b.weight',
		'decoder.encoders.3.norm1.b.bias',
		'decoder.encoders.3.norm2.w.weight',
		'decoder.encoders.3.norm2.w.bias',
		'decoder.encoders.3.norm2.b.weight',
		'decoder.encoders.3.norm2.b.bias',
		'decoder.encoders.4.self_attn.linear_q.weight',
		'decoder.encoders.4.self_attn.linear_q.bias',
		'decoder.encoders.4.self_attn.linear_k.weight',
		'decoder.encoders.4.self_attn.linear_k.bias',
		'decoder.encoders.4.self_attn.linear_v.weight',
		'decoder.encoders.4.self_attn.linear_v.bias',
		'decoder.encoders.4.self_attn.linear_out.weight',
		'decoder.encoders.4.self_attn.linear_out.bias',
		'decoder.encoders.4.feed_forward.w_1.weight',
		'decoder.encoders.4.feed_forward.w_1.bias',
		'decoder.encoders.4.feed_forward.w_2.weight',
		'decoder.encoders.4.feed_forward.w_2.bias',
		'decoder.encoders.4.norm1.w.weight',
		'decoder.encoders.4.norm1.w.bias',
		'decoder.encoders.4.norm1.b.weight',
		'decoder.encoders.4.norm1.b.bias',
		'decoder.encoders.4.norm2.w.weight',
		'decoder.encoders.4.norm2.w.bias',
		'decoder.encoders.4.norm2.b.weight',
		'decoder.encoders.4.norm2.b.bias',
		'decoder.encoders.5.self_attn.linear_q.weight',
		'decoder.encoders.5.self_attn.linear_q.bias',
		'decoder.encoders.5.self_attn.linear_k.weight',
		'decoder.encoders.5.self_attn.linear_k.bias',
		'decoder.encoders.5.self_attn.linear_v.weight',
		'decoder.encoders.5.self_attn.linear_v.bias',
		'decoder.encoders.5.self_attn.linear_out.weight',
		'decoder.encoders.5.self_attn.linear_out.bias',
		'decoder.encoders.5.feed_forward.w_1.weight',
		'decoder.encoders.5.feed_forward.w_1.bias',
		'decoder.encoders.5.feed_forward.w_2.weight',
		'decoder.encoders.5.feed_forward.w_2.bias',
		'decoder.encoders.5.norm1.w.weight',
		'decoder.encoders.5.norm1.w.bias',
		'decoder.encoders.5.norm1.b.weight',
		'decoder.encoders.5.norm1.b.bias',
		'decoder.encoders.5.norm2.w.weight',
		'decoder.encoders.5.norm2.w.bias',
		'decoder.encoders.5.norm2.b.weight',
		'decoder.encoders.5.norm2.b.bias',
		'decoder.after_norm.w.weight',
		'decoder.after_norm.w.bias',
		'decoder.after_norm.b.weight',
		'decoder.after_norm.b.bias',
		'feat_out.weight',
		'feat_out.bias',
		'postnet.postnet.0.0.weight',
		'postnet.postnet.0.1.weight',
		'postnet.postnet.0.1.bias',
		'postnet.postnet.0.1.running_mean',
		'postnet.postnet.0.1.running_var',
		'postnet.postnet.0.1.num_batches_tracked',
		'postnet.postnet.1.0.weight',
		'postnet.postnet.1.1.weight',
		'postnet.postnet.1.1.bias',
		'postnet.postnet.1.1.running_mean',
		'postnet.postnet.1.1.running_var',
		'postnet.postnet.1.1.num_batches_tracked',
		'postnet.postnet.2.0.weight',
		'postnet.postnet.2.1.weight',
		'postnet.postnet.2.1.bias',
		'postnet.postnet.2.1.running_mean',
		'postnet.postnet.2.1.running_var',
		'postnet.postnet.2.1.num_batches_tracked',
		'postnet.postnet.3.0.weight',
		'postnet.postnet.3.1.weight',
		'postnet.postnet.3.1.bias',
		'postnet.postnet.3.1.running_mean',
		'postnet.postnet.3.1.running_var',
		'postnet.postnet.3.1.num_batches_tracked',
		'postnet.postnet.4.0.weight',
		'postnet.postnet.4.1.weight',
		'postnet.postnet.4.1.bias',
		'postnet.postnet.4.1.running_mean',
		'postnet.postnet.4.1.running_var',
		'postnet.postnet.4.1.num_batches_tracked'
		]