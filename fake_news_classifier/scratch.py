
vectorized_df = pd.read_pickle(os.path.join(CURRENT_DIR, VECTORIZED_DATA_PICKLE))

max_seq_len = 500
embedding_size = 100
num_input_tokens = 300

claims = vectorized_df.loc[:, 'claim']
supports = vectorized_df.loc[:, 'support']
labels = vectorized_df.loc[:, 'label']

labels = to_categorical(labels, num_classes=3)

new_claims = []
for claim in claims:
    words_in_claim = []
    for word in claim:
        assert len(word) == 300
        words_in_claim.append(word)
    new_claims.append(words_in_claim)
new_claims = pad_sequences(new_claims, maxlen=max_seq_len, truncating='post', padding='post', dtype='float32')

new_claims = np.array(new_claims)
print(new_claims)
print(new_claims[0].shape)

# supports = [support if len(support) < max_seq_len else support[0:max_seq_len-1] for support in supports]
# # claims = pad_sequences(claims, maxlen=max_seq_len)
# # supports = pad_sequences(supports, maxlen=max_seq_len)

#
# articles_input_model = Sequential()
# articles_input_model.add(Embedding(input_dim=num_input_tokens, output_dim=embedding_size))
# # articles_input_model.add(SpatialDropout1D(0.2))
# articles_input_model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
# articles_input_model.add(Dense(16, activation='relu'))
#
claims_input_model = Sequential()
claims_input_model.add(LSTM(units=64, input_shape=(500, 300)))
claims_input_model.add(Dense(16, activation='relu'))
#
# # mergedOut = Concatenate()([claims_input_model.output,articles_input_model.output])
# # mergedOut = Dense(8, activation='relu')(mergedOut)
# # mergedOut = Dense(3, activation='softmax')(mergedOut)
# #
# # complete_model = Model([claims_input_model.input, articles_input_model.input], mergedOut)
# # complete_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # complete_model.summary()
# #
# # complete_model.fit([claims, supports], labels, batch_size=64, epochs=10, verbose=1, validation_split=0.2)
#
claims_input_model.add(Dense(3, activation='softmax'))
claims_input_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
claims_input_model.summary()
claims_input_model.fit(new_claims, labels, batch_size=64, epochs=10, verbose=1, validation_split=0.2)