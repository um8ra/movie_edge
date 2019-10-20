from django import forms

class SentimentForm(forms.Form):
    likes = forms.CharField(label='Comma separated movieId of liked movieIds')
    dislikes = forms.CharField(label='Comma separated movieId of disliked movieIds')
