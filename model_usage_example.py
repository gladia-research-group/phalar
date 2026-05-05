from contrastive_model.contrastive_model import ContrastiveAudioModelPLWrapper
import torch
import time

# Enable global torch inference mode
with torch.inference_mode():

    # PHALAR can be used like so
    model_path = 'ckpts/PHALAR_best.ckpt'

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )


    model = ContrastiveAudioModelPLWrapper(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # Makes the similarity commutative, I know I should've written a function for this, sorry :(
    model.similarity.hermitian = True

    print("PHALAR Model size:", sum(p.numel() for p in model.parameters()))  # Print number of parameters
    start_time = time.time()
    jibberish_input = torch.randn(1, 1, 5*16000)  # Example input tensor, expects 16kHz audio
    embedding = model.encoder(jibberish_input)
    print("PHALAR Embedding shape:", embedding.shape)
    print("PHALAR Encoding time:", time.time() - start_time, "seconds")

    another_input = torch.randn(1, 1, 7*16000)  # Another example input tensor, can be of different length
    another_embedding = model.encoder(another_input)

    similarity_score = model.similarity(embedding, another_embedding)
    similarity_score_2 = model.similarity(another_embedding, embedding)
    assert torch.allclose(similarity_score, similarity_score_2), "Err, something's wrong, it should be hermitian!"
    print("PHALAR Similarity score:", similarity_score)
    print('')


    # COCOLA can be used like so
    from feature_extraction.feature_extraction import FeatureExtractor
    from contrastive_model import constants

    model_path = 'ckpts/COCOLA_best.ckpt'

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # It has this specific feature extractor
    feat_extractor =  FeatureExtractor(
        feature_extractor_type=constants.ModelFeatureExtractorType.HPSS,
        n_mels=64)


    model = ContrastiveAudioModelPLWrapper(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    print("COCOLA Model size:", sum(p.numel() for p in model.parameters()))  # Print number of parameters

    start_time = time.time()
    jibberish_input = torch.randn(1, 5*16000)  # Example input tensor
    hpss_jibberish = feat_extractor(jibberish_input)[None]
    embedding = model.encoder(hpss_jibberish)
    print("COCOLA Embedding shape:", embedding.shape)
    print("COCOLA Encoding time:", time.time() - start_time, "seconds")

    another_input = torch.randn(1, 7*16000)  # Another example input tensor, can be of different length
    hpss_another_input = feat_extractor(another_input)[None]
    another_embedding = model.encoder(hpss_another_input)

    similarity_score = model.similarity(embedding, another_embedding)
    print("COCOLA Similarity score:", similarity_score)

    # On a note, the original COCOLA score is not symmetric!
    # If you wish you can make it so by doing
    similarity_score_2 = model.similarity(another_embedding, embedding)
    avg_similarity = (similarity_score + similarity_score_2.T) / 2 # of course the .T in this case is trivial since both are 1x1
    print("COCOLA Symmetrized Similarity score:", avg_similarity)