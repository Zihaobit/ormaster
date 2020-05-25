--- 
layout: post 
title: "Elasticsearch" 
date: 2018-05-17 00:00:04 
categories: liuqianchao update
---

## Elasticsearch Tutorial

Lucene是Apache下的文本搜索引擎，而Elasticsearch是以lucene作为内核的，JSON based, distributed 网络服务。因此可以把Elasticsearch看作lucene封装后的分析引擎。

### Easticsearch 基本概念

Document: 文档，以key-value的形式存储。Lucene把文档下不同类型(string, integer, date等)都看作是bytes。   
Index索引：相当于database。
Type类型: 描述同一类document, 相当于表名。
Mapping: 数据库的schema, 记录Type的字段、属性一集字段的数据类型、索引方式

Elasticsearch的查询语言称为Query DSL(Query Domin-Specific-languasge)

**Elasticsearch的文档索引方式**

- 倒排索引(inverted-index): 对于所有文档unique的词汇，维护一个出现文档的列表。从而当查询某几个词汇时，根据文档包含各词汇的数目，按相关性排序并返回。形式：word1: doc1, doc2; word2: doc1; word3: doc1, doc3.


- Doc values: 当使用aggs用terms对某一文本field进行分桶计数时，倒排索引需要对每个word进行group by计数，效率十分低，因此一般采用doc values, 具体而言就是对倒排索引进行转置。doc1： word1, word2, word3; doc2: word4, word5. 这时计算唯一word(或者叫做token)的计数就更方便。和倒排索引一样，Doc, values一样默认对所有字段(数字、经纬度、日期、IP、not_analyzed字符串)开启。对于analysed的字符串，如果要使用terms桶，需要将该字段设置成fielddata，该操作将会对analysed字段生成doc values并存储在内存中，因此要谨慎该操作。更谨慎的做法是将该字段设置为not_analysed。

**Elasticsearch数据类型**

可以使用`GET indexname/_mapping`获取mapping信息, 完整的数据类型见[docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html)

- string: 包括text和keyword两种类型。前者一般用于长文本，是analyzed。后者一般用于类似于email地址，hostnames, status code, tags这种的短文本。但默认(?)情况下，一个字段如果是text类型的，则包含一个feilds的属性, 以支持子字段`keyword`, 该子字段的type是keyword, 因此同一个字段支持多种方式的索引方式。



### Elasticsearch Put

使用put插入数据, 下面向longhash/employee表插入_id为1的document。

```sql

curl -XPUT 'localhost:9200/longhash/employee/1' -d '
{
    "first_name": "qianchao",
    "last_name": "liu",
    "age": "24",
    "interests": ["programming", "sport"]
}

'
```


### Elasticsearch Query Language

检索数据库的语法格式：

```sql
curl -XGET 'localhost:9200/longhash/employee/_search' -d '
{
    "query":{
        "match":{
            "last_name": "liu"
        }
    },
    "highlight": {
        "fields": {
            "age":{}
        }
    }
}


'

```

#### from, size, 子列约束 

通过from设置从第几个结果开始
size则相当用limit
fileds:["subcolumn"]用来约束要返回的子列, 但默认情况下fileds的支持是不开启的，官方更建议使用“_source”:["subcolumn"]

```sql
{
  "from": 0,
  "size": 3,
  "_source": ["message"],
  "query": {
    "query_string": {
      "query": "message:bitcoin"
    }
  }
}
```

#### term

term要求完全匹配(and关系)传入的值。

term: 单条查询

```sql
{
    "query":{
        "term"：{
            "message": "bitcoin"
        }
    }
}
```

terms: 多条查询, 通过minimum_match确定至少符合条件的数目

```sql
{
    "query":{
        "term"：{
            "message": ["bitcoin", "ethereum"],
            "minimum_match":1
        }
    }
}

```

#### match查询

不同于term的完全匹配，match默认对字符串是分词后全文搜索Query(对数字、时间、bool或者not_analyzed字段仍是精确匹配Filter），并根据tf-idf值排序后返回结果, 可以使用operator该修改or的逻辑：
match使用operator来设置匹配模式：and/or

```sql
{
  "query": {
    "match": {
      "message": {
        "query": "keyword1 keyword2",
        "operator": "and"
      }
    }
  }
}
```

使用slop匹配缺失值, slop为缺省内容的数目, 下面的语句会返回所有keyword1 xxx xxx keyword4

```sql
{
  "query": {
    "match": {
      "message": {
        "query": "keyword1 keyword4",
        "slop": 2
      }
    }
  }
}
```

使用multi_match，来对不同的字段，使用相同的规则, 只要有一个字段满足条件就会返回数据

```sql
{
    "multi_match":{
        "query" : "bitcoin"
        "fields": ["title", "body"]
    }
}
```


#### bool

所有符合条件都需要使用bool语句

- "must":  必须匹配
- "should":  至少满足一个
- "must_not": 不满足
- "filter": 必须匹配，但不评分

```python
{
    "query":{
        "filtered":{
            "filter" :{
                
                    "bool":{
                        "must":{
                            "term":{
                                "message": "bitcoin"
                                }
                        },
                        "should": {
                            "range":{
                                "timestamp": {"gte": 15, "lte": 20}
                            }
                        },
                        "must_not": {
                            "term": {
                                "message": "eos"
                            }
                        }
                    }

            }
        }




    }
}
```


#### exists, missing

相当于sql中的is not null和is null

```sql
{
    "exists": {
        "field": "title"
    }
}

```

#### filter

上面我们写了很多条件筛选的语句，那么为什么这还需要`filter`关键词？这里是因为非filter的查询一般会进行doc评分，所以速度回比较慢，而一旦指定了filter查询，则进行精确查询。

例1， 使用term，将默认进行文档评分
```sql
{
    "query"：{
        "term": {"price":20}
    }
}
```

例2，使用constant_score，取消文档评分, 常用来代替只有filter的bool语句

```sql
{
    "constant_score":{
        "filter": {
            "term": {"price":20}
        }
    }
}
```

例3， 但当使用constan_score：filter: term查询文本时，如果文本字段是analyzed时，使用term会抹去大小写，符号等，从而无法实现精确值查询，为避免该问题，需要将字段index设置为not_analyzed.


#### sort

```sql
 "query":{
     "bool":{
         "must": {
             "match":{
                 "tweet": "some text"
                 }},
         "filter":{
             "term":{
                 "user_id":1
             }
         }
     }
 }

 "sort": [
    {
      "@timestamp": "desc"
    }, {
      "_score": "desc"
    }
  ]

```


#### aggregation

buckets: 使用group by将文档划分成符合不同条件的子集。

metrics：count(), sum(), max()等计算。

查询各个颜色产品的数量。这里使用bucket的方式是terms, 其他分桶的方法包括histogram(field, interval), date_histogram(filed, interval) 。但上述两种histogram默认只返回文档数目非0的桶，如果想要x axis连续，需要设置两个额外的参数："min_doc_count":0, "extended_bounds": {"min": "2018-05-01", "max":"2018-06-01"}

```sql
{
    "size":0, // 只返回aggs的结果。
    "aggs": {
        "name_your_bucket":{
            "terms":{
                "filed": "color" // 这里是一个terms桶，每个color动态创建一个桶。
            }
        }
    }
}

```


查询各个颜色产品的的平均价格，这里使用的metrics是avg, 其他常用的包括max, min, sum, extended_stats, percentiles等：
```sql
{
    "size":0,
    "aggs": {
        "name_your_bucket":{
            "terms":{ // 这里是一个terms桶，每个color动态创建一个桶。
                "filed": "color" 
            },
            "aggs":{
                "name_your_metrics":{
                    "avg":{ // 这里是一个avg度量
                        "field":"price"
                    }
                }
            }
        }
    }
}

```


上面仅仅是进行聚合，如果想要加入筛选需要指定query的内容。上面的两个查询等价于在query里使用match_all。

```sql
{
    "size":0,
    "query": {
        "bool":{
            "filter": {
                "year": 2018
            } 
        }
    },
    "aggs":{
        "name_your_bucket":{
            "terms":{
                "field": "color"
            }
        }
    }
}

```

使用`cardinality`实现`distinct`，相当于`select distinct(color) from table`

```sql
{
    "size":0,
    "aggs":{
        "name_your_result":{
            "cardinality":{
                "field": "color“
            }
        }
    }
}

```