import gensim
import pandas as pd
from ml.processing import TextNormalizer


# Modelling artifacts
data_path = 'artifacts/'


def load_model_artifacts(country: str, model_name: str):
    dictionary = gensim.corpora.Dictionary.load(data_path + country + '/' + model_name + '/' + 'dictionary.dict')
    
    if model_name == 'lda':
        model = gensim.models.LdaModel.load(data_path + country +  '/' + model_name + '/' + 'model.' + model_name)
    
    elif model_name == 'tfidf':
        model = gensim.models.TfidfModel.load(data_path + country +  '/' + model_name + '/' + 'model.' + model_name)

    index = gensim.similarities.MatrixSimilarity.load(data_path  + country + '/' + model_name + '/' + 'queries_index.index')
    return dictionary, model, index
    
# LDA
zanz_dictionary, zanz_lda_model, zanz_lda_index = load_model_artifacts('zanzibar', 'lda')
mald_dictionary, mald_lda_model, mald_lda_index = load_model_artifacts('maldives', 'lda')

# TFIDF
zanz_dictionary, zanz_tfidf_model, zanz_tfidf_index = load_model_artifacts('zanzibar', 'tfidf')
mald_dictionary, mald_tfidf_model, mald_tfidf_index = load_model_artifacts('maldives', 'tfidf')

# DOC2VEC
zanzibar_doc2vec = gensim.models.doc2vec.Doc2Vec.load(data_path + 'zanzibar/doc2vec/model.doc2vec')
maldives_doc2vec = gensim.models.doc2vec.Doc2Vec.load(data_path + 'maldives/doc2vec/model.doc2vec')

# Dataset for reference
zanz_data = pd.read_excel(data_path + 'zanzibar/data.xlsx')
mald_data = pd.read_excel(data_path + 'maldives/data.xlsx')

normalizer = TextNormalizer()


def sort_ids(sims, n_match: int):
    sorted_ids = []
    similarities = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, sim in similarities[:n_match]:
        sorted_ids.append(i)
    return sorted_ids


def search_similar(query: str, n_match: int, model:str, country: str):
    
    clean_query = normalizer.clean(query)

    if model == 'LDA':
        
        if country == 'Maldives':
            bow_query = mald_dictionary.doc2bow(clean_query)
            lda_query = mald_lda_model[bow_query]
            sims = mald_lda_index[lda_query]
        
        elif country == 'Zanzibar':
            bow_query = zanz_dictionary.doc2bow(clean_query)
            lda_query = zanz_lda_model[bow_query]
            sims = zanz_lda_index[lda_query]
        
        sorted_ids = sort_ids(sims, n_match)

    elif model == 'Tf-Idf':
        
        if country == 'Maldives':
            bow_query = mald_dictionary.doc2bow(clean_query)
            tfidf_query = mald_tfidf_model[bow_query]
            sims = mald_tfidf_index[tfidf_query]
        
        elif country == 'Zanzibar':
            bow_query = zanz_dictionary.doc2bow(clean_query)
            tfidf_query = zanz_tfidf_model[bow_query]
            sims = zanz_tfidf_index[tfidf_query]
        
        sorted_ids = sort_ids(sims, n_match)
    
    elif model == 'Doc2Vec':

        if country == 'Maldives':
            vec = maldives_doc2vec.infer_vector(clean_query)
            sims = maldives_doc2vec.docvecs.most_similar([vec], topn=n_match)
        
        elif country == 'Zanzibar':
            vec = zanzibar_doc2vec.infer_vector(clean_query)
            sims = zanzibar_doc2vec.docvecs.most_similar([vec], topn=n_match)

        sorted_ids = [idx for (idx, sim) in sims]

    return sorted_ids


def query(query: str, n_match: int, model: str, country: str):

    ids = search_similar(query, n_match, model, country)
    
    if country == 'Zanzibar':
        data = zanz_data.copy()
    
    elif country == 'Maldives':
        data = mald_data.copy()

    result_df = data.loc[ids, ['name', 'url', 'rating']]
    result_df.rename(
        columns={'name': 'Name', 'url': 'URL', 'rating': 'Rating'},
        inplace=True
    )
    result_df.reset_index(
        inplace=True,
        drop=True
    )
    result_df.index = result_df.index + 1
    return result_df

