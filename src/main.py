from model import Model

opt = Model()
text = " Hey guys welcome back to another youtube video! Today we are going to be doing some more card pack openings for Magic The Gathering and I hope that we are going to get lucky and finally get the brand new golden legendary card that is all the rage,"

outputs = opt.get_text_activations( text )
input, attn, ff, output = outputs

print( opt.predict( text ) )