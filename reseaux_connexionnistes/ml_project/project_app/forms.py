from django import forms

class ToxicityForm(forms.Form):
    cico = forms.FloatField(label='cico')
    sm1_dz = forms.FloatField(label='SM1_Dz(Z)')
    gats1i = forms.FloatField(label='GATS1i')
    ndsch = forms.IntegerField(label='NdsCH')
    ndssc = forms.IntegerField(label='NdssC')
    mlogp = forms.FloatField(label='MLOGP')