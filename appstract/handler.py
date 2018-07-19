import simplejson as json
import requests
import untangle


def getAbs(q):
    try:
        return q.PubmedArticleSet.PubmedArticle.MedlineCitation.Article.Abstract.AbstractText.cdata
    except AttributeError:
        try:
            return "\n".join([p.cdata for p in q.PubmedArticleSet.PubmedArticle.MedlineCitation.Article.Abstract.AbstractText])
        except AttributeError:
            return ""


def getDOI(q):
    doi = ""

    try:
        doi_obj = q.PubmedArticleSet.PubmedArticle.MedlineCitation.Article.ELocationID
        try:
            if doi_obj['EIdType'] == "doi":
                doi = doi_obj.cdata
        except TypeError:
            dois = [d.cdata for d in doi_obj if d["EIdType"] == "doi"]
            if (len(dois)):
                doi = dois[0]
    except AttributeError:
        pass

    return doi


def getYear(q):
    # sometimes the year info isn't provided :(
    try:
        year = q.PubmedArticleSet.PubmedArticle.MedlineCitation.Article.Journal.JournalIssue.PubDate.Year.cdata
    except AttributeError:
        year = ''
    return year


def get_abstract(pmid):
    """
    Steps:

    1. query pubmed w/ the pmid
    2. extract doi, date, and abstract text
    3. query for an open access version of the article,
    if it exists, return link to PDF of article.
    """
    pubmed_template = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.\
fcgi?db=pubmed&id={}&tool=appstract&email=keshavan@berkeley.edu&retmode=xml"
    d = requests.get(pubmed_template.format(pmid))
    with open("{}.xml".format(pmid), "w") as f:
        f.write(d.text)

    q = untangle.parse("{}.xml".format(pmid))
    abstract = getAbs(q)
    doi = getDOI(q)
    year = getYear(q)

    oab_url_template = "https://api.openaccessbutton.org/availability/\
?apikey=91ec4d7e0a8ac9bb4d5ecdaf3424d8&doi={}"
    unpaywall_template = "https://api.unpaywall.org/v2/{doi}?\
email=keshavan@berkeley.edu"

    unpaywall = None

    if (doi != ""):
        # oab = requests.get(oab_url_template.format(doi))
        unpaywall = requests.get(unpaywall_template.format(doi=doi)).json()

    return dict(abstract=abstract,
                doi=doi,
                year=year,
                open_access=unpaywall)


def handle(st):
    # inp = json.loads(st)
    print(json.dumps(get_abstract(st)))
