from deeppavlov import build_model, configs



model = build_model('levenshtein_corrector_ru', download=True)