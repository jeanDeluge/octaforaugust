from flask import Blueprint, render_template, url_for
from werkzeug.utils import redirect # http 요청에 필요
# from sophia.models import 나중에 모델 들어올 자리

bp = Blueprint('main', __name__, url_prefix='/')

#  메인 페이지
@bp.route('/')
def main():

    return render_template('index.html')