from django import forms
from cse6242_team5.settings import BASE_DIR
from pathlib import Path
import os

base_path = Path(BASE_DIR)
gensim_path = base_path / '..' / 'gensim_models2'

class SentimentForm(forms.Form):
    gensim_model = forms.ChoiceField(choices=[tuple([str(i).split(os.path.sep)[-1]] * 2) for i in gensim_path.iterdir()])
    likes = forms.CharField(label='Comma separated movieId of liked movieIds')
    dislikes = forms.CharField(label='Comma separated movieId of disliked movieIds')
