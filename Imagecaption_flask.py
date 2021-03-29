from flask import Flask,render_template,redirect,request
import Caption
app=Flask(__name__)
@app.route('/')
def home():
    
    return render_template("ImageCaptioning.html")
@app.route('/',methods=['POST'])
def Image():
    
    if request.method=='POST':
        f=request.files['userfile']
        path="./static/{}".format(f.filename)
        f.save(path)
        captions=Caption.caption_this_image(path)
        result_dic={
'image':path,
'caption':captions
            }
    
    return render_template('ImageCaptioning.html',your_result=result_dic)
if __name__=='__main__':
    app.run(debug=True)
