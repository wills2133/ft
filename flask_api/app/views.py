# -*- coding: utf-8 -*-
from app import app
from flask import render_template, request, jsonify
from flask import Response
from .model import diagnose_server as diagnose
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from datetime import datetime
import os
import shutil
import json

fdd = diagnose.diagnose_face_data() #face_diagnose_data
tdd = diagnose.diagnose_tongue_data() #tongue_diagnose_data
lc = diagnose.leancloud_jobs()
online = 1
save_caches = 1
load_caches = 1
# fdd.collect_data()
# fdd.collect_reult()
# tdd.collect_data()
# tdd.collect_reult()

class cache_class():
    def __init__(self):
        self.cache = {}
        self.caches = {}
        self.source_dir = './app/resource'
        self.cache_dir = ''
        self.cache_index_dir = './app/caches/index.json'
        self.cell = ''
        self.cache_exist = False
    def init_cache(self, cell, name):
        self.cache = {}
        self.caches = {}
        self.cell = name+'-'+str(cell)
        self.cache_dir = './app/caches/' + self.cell
    def save_pic(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        shutil.copytree(self.source_dir, self.cache_dir)
    def load_pic(self):
        if os.path.exists(self.cache_dir) and os.path.exists(self.source_dir):
            shutil.rmtree(self.source_dir)
            shutil.copytree(self.cache_dir, self.source_dir)
            print('load pic')
    def add_check_save(self, name, content):
        self.cache[name] = content
        if len(self.cache.keys()) == 8:
            with open(self.cache_dir + '/responses.json', 'w') as fp:
                json.dump(self.cache, fp)
                print('save caches to ' + self.cache_dir + '/responses.json')
    def check_load(self):
        if os.path.exists(self.cache_dir + '/responses.json'):
            print ("exists" + self.cache_dir + '/responses.json')
            with open(self.cache_dir + '/responses.json', 'r') as fp2:
                self.caches = json.load(fp2)
            print('load caches ' + self.cache_dir + '/responses.json')
            return True
        else:
            return False

cc = cache_class()

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './app/caches'
configure_uploads(app, photos)

@app.route('/receive', methods=['GET', 'POST'])
def receive():
    message = {}
    if request.method == 'POST' and 'photo' in request.files:
        name = request.form['name']
        cell = request.form['cell']
        print('receive photo name: ' + name)
        print('receive photo cell: ' + cell)
        # remove folder if exists
        if os.path.exists('./app/caches/'+ name+'-'+cell):
            shutil.rmtree('./app/caches/'+ name+'-'+cell)
        # save picture
        filename = photos.save(request.files['photo'], name+'-'+cell, '1.jpg')
        filename2 = photos.save(request.files['photo2'], name+'-'+cell, '2.jpg')
        face_url = photos.url(filename)
        tongue_url = photos.url(filename2)
        print('received photos')
        message = {'furl':face_url, 'turl':tongue_url}
    else:
        message = {'furl':'fail_url', 'turl':'fail_url'}
    return jsonify(status="success", msg=message)

# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route("/face-tongue")
def face_tongue_page():
    return render_template("face-tongue.html")

@app.route("/process-pic/<user_name>/<user_id>", methods=["GET"])
def process_pic(user_id, user_name):
    print (user_id)
    print (user_name)
    message = ""
    if online:
        caches_loaded = False
        if load_caches:
            cc.init_cache(user_id, user_name)
            if cc.check_load():
                print ("check_load")
                cc.load_pic()
                fdd.face_res = cc.caches['result']['face']
                tdd.face_res = cc.caches['result']['tongue']
                fdd.hist_data = cc.caches['hist']
                fdd.organ_position = cc.caches['fpos']
                tdd.organ_position = cc.caches['tpos']
                fdd.reulst_prob = cc.caches['donut']
                tdd.polar_res = cc.caches['tpolar']
                fdd.polar_res = cc.caches['fpolar']
                fdd.organ_res= cc.caches['organ_res']
                tdd.organ_position['tongue'] = fdd.organ_position['tongue']
                tdd.reulst_prob['tongue'] = fdd.reulst_prob['tongue']
                tdd.organ_res['tongue'] = fdd.organ_res['tongue']
                tdd.hist_data['tongue'] = fdd.hist_data['tongue']
                message = "done"
                caches_loaded = True
        if not caches_loaded:
        #     'Asia/Shanghai' 'America/Chicago'
            na_date = lc.get_date('Canada/Pacific')
            registered = lc.query_register_info('patient_info', 'name', user_name, 'cell', user_id, 'birthDate', 'gender', na_date)
            if registered:
                photographed = lc.query_pic_url('patient_current_info', 'name', user_name, 
                    'cell', user_id, 'currentdate', na_date, 'facialpictureURL', 'tonguepictureURL', 'objectId')
                if photographed:
                    ## if photo is upload to local server
                    cc.init_cache(user_id, user_name)
                    cc.load_pic()
                    downloaded = True
                    ## if photo is upload to leancloud
                    # downloaded = lc.download_with_timeout(10)
                    if downloaded:
                        try:
                            fdd.collect_data()
                            tdd.collect_data()
                            with open('./app/model/map_patients.json', 'r') as fp:
                                map_patients = json.load(fp)
                            if user_id in map_patients.keys():
                                print("found in map_patients")
                                fdd.collect_reult(lc.age_category, lc.gender, map_patients[user_id][1])
                                tdd.collect_reult(lc.age_category, lc.gender, map_patients[user_id][0])
                            else:
                                fdd.collect_reult(lc.age_category, lc.gender)
                                tdd.collect_reult(lc.age_category, lc.gender)
                            message = "done"
                        except Exception as error:
                            print(repr(error))
                            message = "wrong_input"
                        if message == "done":
                            if save_caches:
                                cc.save_pic()
                        lc.update_results('patient_current_info', 'facialdiagnosiscomputer', 
                                      'tonguediagnosiscomputer', fdd.face_res[0].split('：')[0], tdd.face_res[0].split('：')[0])
                    else:
                        message = "not_downloaded"
                else:
                    message = "not_photographed"
            else:
                message = "not_registered"
    else:
        fdd.collect_data()
        fdd.collect_reult()
        tdd.collect_data()
        tdd.collect_reult()
        if save_caches:
            cc.init_cache(user_id)
            cc.save_pic()
        message = "done"
    
    return jsonify(status="success", msg=message)

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/image/<imageid>")
def response_pic(imageid):
    image = open("./app/resource/{}.jpg".format(imageid), 'rb')
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route("/contour-path", methods=["GET"])
def contour_api():
    contour_paths = ['M 22,110 Q 22,119, 22,128 Q 23,136, 24,144 Q 26,152, 28,160 Q 32,167, 36,174 Q 40,181, 44,188 Q 50,193, 54,198 Q 60,203, 64,208 Q 71,209, 78,210 Q 85,210, 92,210 Q 98,206, 104,202 Q 110,196, 116,192 Q 122,186, 128,180 Q 132,172, 136,165 Q 140,157, 142,148 Q 144,140, 146,130 Q 146,121, 146,112 ', 'M 40,111 Q 44,110, 48,108 Q 52,108, 56,109 Q 59,112, 62,114 ', 'M 62,114 Q 59,114, 55,115 Q 51,115, 46,114 Q 43,113, 40,111 ', 'M 101,115 Q 105,112, 108,110 Q 113,110, 117,110 Q 120,112, 124,113 ', 'M 124,113 Q 120,115, 117,116 Q 113,116, 108,116 Q 105,116, 101,115 ', 'M 59,181 Q 62,180, 66,178 Q 69,177, 72,176 Q 76,178, 78,178 Q 81,178, 84,177 Q 88,178, 92,180 Q 96,181, 100,183 ', 'M 100,183 Q 96,186, 92,188 Q 88,190, 84,191 Q 81,191, 78,192 Q 75,191, 72,190 Q 68,189, 65,187 Q 62,184, 59,181 ', 'M 59,181 Q 61,181, 62,182 Q 68,181, 72,181 Q 75,182, 78,182 Q 81,182, 84,182 Q 92,182, 100,183 ', 'M 81,113 Q 80,120, 80,128 Q 80,135, 80,142 Q 79,150, 78,157 ', 'M 64,159 Q 68,161, 71,162 Q 75,164, 79,166 Q 83,164, 88,163 Q 91,162, 95,160 ']
    return jsonify(status="success", msg=contour_paths)

@app.route("/result", methods=["GET"])
def result_api():
    result = {
        'face':fdd.face_res,
        'tongue':tdd.face_res,
    }
    if save_caches:
        cc.add_check_save('result', result)
    return jsonify(status="success", msg=result)

@app.route("/hist-data", methods=["GET"])
def hist_organ_api():
    fdd.hist_data['tongue'] = tdd.hist_data['tongue']
    if save_caches:
        cc.add_check_save('hist', fdd.hist_data)
    return jsonify(status="success", msg=fdd.hist_data)

@app.route("/organ-position", methods=["GET"])
def organ_position_api():
    fdd.organ_position['tongue'] = tdd.organ_position['tongue']
    if save_caches:
        cc.add_check_save('fpos', fdd.organ_position)
    return jsonify(status="success", msg=fdd.organ_position)

@app.route("/tongue-position", methods=["GET"])
def tongue_position_api():
    if save_caches:
        cc.add_check_save('tpos', tdd.organ_position)
    return jsonify(status="success", msg=tdd.organ_position)

@app.route("/result-color", methods=["GET"])
def color_result_api():
    if save_caches:
        cc.add_check_save('fpolar', fdd.polar_res)
    return jsonify(status="success", msg=fdd.polar_res)

@app.route("/result-tongue-color", methods=["GET"])
def color_tongue_result_api():
    if save_caches:
        cc.add_check_save('tpolar', tdd.polar_res)
    return jsonify(status="success", msg=tdd.polar_res)

@app.route("/donut-data", methods=["GET"])
def donut_data_api():
    fdd.reulst_prob['tongue'] = tdd.reulst_prob['tongue']
    if save_caches:
        cc.add_check_save('donut', fdd.reulst_prob)
    return jsonify(status="success", msg=fdd.reulst_prob)

@app.route("/result-organ", methods=["GET"]) 
def result_organ_api(): 
    fdd.organ_res['tongue'] = tdd.organ_res['tongue']
    if save_caches:
        cc.add_check_save('organ_res', fdd.organ_res)
    return jsonify(status="success", msg=fdd.organ_res)
