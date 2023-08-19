from flask import Blueprint, jsonify, render_template, request, url_for
from werkzeug.utils import redirect
from ..apis.sophia_for_user import SophiaForUser

# from sophia.models import
# from sophia.forms import

bp = Blueprint('user', __name__, url_prefix='/user')

# 도서 위치 찾기
@bp.route('/find_book', methods=['GET'])
def find_book():
    book = request.args.get('book')
    # value 값을 book으로 받기
    response_data = {
        'message' : 'GET request received',
        'book' : book
    }
    # 데이터 베이스 검색

    return render_template('user/user_find.html')


# 책 추천받기
@bp.route('/recommend_book', methods=['GET'])
def recommend_book():
    with SophiaForUser(request=request).recommend_book() as book_list:
        return book_list

