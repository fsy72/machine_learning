from django import forms

class ToxicityForm(forms.Form):
    cic0 = forms.FloatField(label='CIC0')
    sm1_dz = forms.FloatField(label='SM1_Dz(Z)')
    gats1i = forms.FloatField(label='GATS1i')
    ndsch = forms.IntegerField(label='NdsCH')
    ndssc = forms.IntegerField(label='NdssC')
    mlogp = forms.FloatField(label='MLOGP')