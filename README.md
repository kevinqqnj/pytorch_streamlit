# pytorch_streamlit

A pretrained model that fakes it until it makes it. 

## run locally
```
pip install -r requirements.txt
streamlit run <your-file>.py --server.runOnSave true
```

## deploy to Heroku
1. create new app in heroku
2. push
```
heroku git:remote -a streamlit-gan
git add .
git commit -m "heroku"
git push heroku master
```

## Note
1. Slug size: use specific torch version instead of full version
`https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-cp36-cp36m-linux_x86_64.whl`

2. Memory quota vastly exceeded
Set heroku app Free Dynos to "Hobby"
