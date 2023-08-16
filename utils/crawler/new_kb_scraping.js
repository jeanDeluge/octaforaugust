`use strict`

const axios = require("axios")
const cheerio = require("cheerio")
const puppeteer = require("puppeteer")
const fs = require('fs');


//지연함수
async function sleep(ms){
    const wakeUpTime = Date.now() + ms;
    while(Date.now() < wakeUpTime){}
}

// Paging_crawling를 통해서 모든 책 href 을 집어넣는다. 
let book_urls = []
let book_obj_arr = []

async function Paging_crawling(href){

    const browser = await puppeteer.launch({
        headless: false,
    })

    const page = await browser.newPage();
    
    await page.setViewport({
        width: 1440,
        height: 1080
    })
    await page.goto(href,
        {
            waitUntil: "load"
        });
    const html = await page.content()
    const $ = cheerio.load(html)
    
    await sleep(3000)
    
    //해당 페이지 내에 있는 모든 책 url 가져오기
    for(let i= 1; i < 21; i++){
        let url = $(`#homeTabAll > div.switch_prod_wrap.view_type_list > ol > li:nth-child(${i}) > div.prod_area.horizontal > div.prod_thumb_box.size_lg > a`).attr("href")
        if(url){
            book_urls.push(url)
        }
        
        
    }

    await browser.close()
}

async function detail_crawling(href){
    
    const browser = await puppeteer.launch({
        headless: false,
    })

    const page = await browser.newPage();

    await page.setViewport({
        width: 1440,
        height: 1080
    })
    await page.goto(href, {
        waitUntil: 'load'

    });
    const html = await page.content()
    const $ = cheerio.load(html)

    let book_detail = {}
    
    //저자
    let authers =[]
    let get_authors = $('#contents > div.prod_detail_header > div > div.prod_detail_view_wrap > div.prod_detail_view_area > div:nth-child(1) > div > div.prod_author_box.auto_overflow_wrap > div.auto_overflow_contents > div > div > a')
    get_authors.map((idx, el)=>{
        authers.push(el.children[0]["data"])
    })

    let title = $('#contents > div.prod_detail_header > div > div.prod_detail_title_wrap > div > div.prod_title_box.auto_overflow_wrap > div.auto_overflow_contents > div > h1 > span').text()
    let score = $('#ReviewList1 > div.klover_review_box > div.klover_box_left > div.box_top > div > div.caption > span > span.val').text()
    
    let kloba_reviews= []
    //0: 집중돼요 1:도움돼요, 2: 쉬웠어요, 3: 최고예요, 4: 추천해요
    for(let i=1; i < 6; i++){
        let kloba_review = $(`#ReviewList1 > div.klover_review_box > div.klover_box_right > div.box_bottom > div > div:nth-child(${i}) > div > span`).text()
        kloba_reviews.push(Number(kloba_review.slice(0,-1)))
    }

    //함께 구매한 책
    let related_books = []
    for(let i=1; i< 6; i++){
        let related_book = $(`#scrollSpyProdInfo > div.product_detail_area.product_related > div.round_gray_box > div > ul > li:nth-child(${i}).prod_item> div > div.prod_info_box > a > span`).text()
        related_books.push(related_book)
    }

    //book isbn
    let isbn = $('#scrollSpyProdInfo > div.product_detail_area.basic_info > div.tbl_row_wrap > table > tbody > tr:nth-child(1) > td').text()
    let published_date = $('#scrollSpyProdInfo > div.product_detail_area.basic_info > div.tbl_row_wrap > table > tbody > tr:nth-child(2) > td').text()
    let book_pages = $('#scrollSpyProdInfo > div.product_detail_area.basic_info > div.tbl_row_wrap > table > tbody > tr:nth-child(3) > td').text()
    // book category - 큰 분류만
    let category = $("#scrollSpyProdInfo > div.product_detail_area.book_intro > div.intro_book > ul > li > a:nth-child(3)").text()
    let review_box = $('#ReviewList1 > div.tab_wrap.type_sm > div.tab_content > div > div.comment_list > div:nth-child(1)')
    let review = $(review_box).find('.comment_text_box').text()
    book_detail["category"] = category
    book_detail["title"] = title
    book_detail["authers"] = authers
    book_detail["score"] = score
    book_detail["kloba_review"] = kloba_reviews
    book_detail["isbn"] = isbn
    book_detail["published_date"] = published_date
    book_detail["book_pages"] = book_pages
    book_detail["review"] = review
    book_obj_arr.push(book_detail)

    browser.close()

}


async function get_books_url(first, last){

    let current_page = first
    let last_page = last
 

    while (current_page < last_page){
        try{
            let page_url = `https://product.kyobobook.co.kr/category/KOR/0701#?page=${current_page}&type=all&per=20&sort=new`
            await Paging_crawling(page_url)
            await sleep(3000)
            current_page +=1
        }catch(e){
            console.error(e)
            await sleep(3000)
        }
    }
    
}

async function book_detail_list(first, last_page){

    let time = Date.now()
    
    for(let i=0; i <book_urls.length ; i++){
        
        try{
            await detail_crawling(book_urls[i])
            await sleep(3000)
        }catch(e){
            console.error(e, Date.now(), "book details_crawling error")
        }
    }

        const jsonData = JSON.stringify(book_obj_arr, null, 2)

        fs.writeFileSync(`${first}_to_${last_page}_page_data.json`, jsonData)
}




// 첫번째 인자 시작페이지, 두번째 인자 마지막페이지 + 1


async function main(first, last){

    await get_books_url(first, last)

    if(book_urls.length > 0){

        await book_detail_list(first, last)
    }
}

main(1, 6)

// book_urls 전체를 순회해서 detail_crawling 을 진행한다.


    // 통계 https://product.kyobobook.co.kr/api/review/statistics?saleCmdtid=S000200746091
    // 연관 책 구하기