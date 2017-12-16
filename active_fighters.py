import scrapy


class ActiveFighters(scrapy.Spider):
    name = 'fighters'
    start_urls = ['https://en.wikipedia.org/wiki/List_of_current_UFC_fighters']

    def parse(self, response):

        tables = [response.xpath('//*[@id="mw-content-text"]/div/table[5]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[6]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[7]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[8]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[9]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[10]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[11]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[12]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[13]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[14]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[15]')[0],
                  response.xpath('//*[@id="mw-content-text"]/div/table[16]')[0]
                  ]

        for table in tables:
            yield{
                'Fighter': table.css('.fn a::text').extract()
            }

