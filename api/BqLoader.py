from google.cloud import bigquery
import json

class BqLoader:
    client = ''
    tableName = ''
    projName = ''
    datasetName = ''
    queryBody=''
    def __init__(self, projName, datasetName, tableName, columms):
        BqLoader.client = bigquery.Client()
        BqLoader.projName = projName
        BqLoader.datasetName = datasetName
        BqLoader.tableName = tableName
        BqLoader.queryBody = """
                                    ##standardSQL
                                    SELECT
                                      {}
                                    FROM
                                      `{}`
                                    """
        self.columns = columms

    @classmethod
    def getPath(cls):
        return cls.projName + '.' + cls.datasetName + '.' + cls.tableName

    def getQuery(self):
        colStr = ''

        for idx, val in enumerate(self.columns):
            colStr += val;
            if idx+1 != len(self.columns):
                colStr += ','
        print('[test] colStr : {}'.format(colStr));

        ret = BqLoader.queryBody.format(colStr, BqLoader.getPath())
        return ret

    def RequestQuery(self, query):
        results = query.result()

        lstRet = []
        for idx, row in enumerate(results):
            # print("result {} : {}".format(idx,row))
            objTmp = {}
            for i, v in enumerate(self.columns):
                objTmp[v] = row[i]
            # print(objTmp)
            lstRet.append(objTmp)

        ret = json.dumps(lstRet)
        return ret

class ResInfoLoader(BqLoader):
    @staticmethod
    def isOkBQStatus():
        return True

    def getQueryWhere(self,city, menuCate):
        return "WHERE city='{}' AND menuCategory='{}'".format(city, menuCate)

    def query_information_from_us_restaurants(self, city, menuCate):
        query = self.client.query(self.getQuery() + self.getQueryWhere(city, menuCate))
        print('[test] query : {}'.format(super(ResInfoLoader, self).getQuery() + self.getQueryWhere(city, menuCate)))
        return self.RequestQuery(query)

class ResCityLoader(BqLoader):
    def query_city_from_us_restaurants(self):
        query = self.client.query(self.getQuery())
        return self.RequestQuery(query)

class ResMenuCateLoader(BqLoader):
    def query_city_from_us_restaurants(self):
        query = self.client.query(self.getQuery())
        return self.RequestQuery(query)
