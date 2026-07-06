import warnings
warnings.filterwarnings('ignore')
import html
import ipywidgets as widgets
import anywidget
import traitlets
import jupyter
from IPython.display import display, Javascript
import voila
from custom_widgets import *

def createTooltip(description):
    source = JUPYTER_FILE_PATH + "info.svg"
    return f"""&nbsp; <img src="{source}"
    style="cursor:pointer;position:absolute;top:0px;"
    title="{description}">"""

class FilterModule():
    def __init__(self):
        self.filterBy = widgets.HTML(value = "Refine Results").add_class("heading5").add_class("medium")
        dateRange = widgets.HTML(value = "Date").add_class("body2").add_class("medium")
        self.fromDate = widgets.DatePicker(description = "From")
        self.toDate = widgets.DatePicker(description = "To")
        self.fromDate.add_class("date-constraint") # The script to set the elements attribute is attached to the toggleSwitch widget
        self.toDate.add_class("date-constraint") # This was done for convenience and should be changed later
        self.weightBy = WeightBy()
        self.dateBox = widgets.VBox([dateRange, widgets.HBox([self.fromDate, self.toDate])]).add_class("date-bar")
        repostsToolTip = createTooltip("When disabled, only original posts will appear in search results")
        self.allowRetweets = ToggleSwitch(label = "Include reposts" + repostsToolTip)
        self.retweets = widgets.VBox([widgets.HTML(value = "Retweets").add_class("body2").add_class("medium"), self.allowRetweets])
        self.geography = SearchBar(header = "Geography", placeholder = "Search")
        self.userName = SearchBar(header = "Username", placeholder = "Search")
        self.mainPageClear = widgets.Button(description='Clear All').add_class("clear-button")
        self.mainPageSearch = widgets.Button(description='Apply').add_class("generic-button").add_class("main-page-search")
        self.popUpOptions = widgets.HBox([self.mainPageClear, self.mainPageSearch]).add_class("pop-up-options")
        self.mainFilters = widgets.VBox([self.filterBy, self.dateBox, self.retweets, self.geography, self.userName, self.weightBy]).add_class("main-filters")
        self.filterBox = widgets.VBox([self.mainFilters]).add_class("filter-box")

class AdvancedPage():
    def __init__(self):
        self.searchButton = widgets.Button(description='Search', icon="search").add_class("generic-button").add_class("search-button")
        self.hiddenButton = widgets.Button()
        self.hiddenButton.add_class("hidden-button") # work around for syncing search when the user still has input in the search bars
        
        self.clearButton = widgets.Button(description='Clear All').add_class("clear-button")
        
        self.bottomBar = widgets.HBox([self.clearButton, self.searchButton, self.hiddenButton], layout = widgets.Layout(justify_content = "flex-end"))
        self.keyWordSearch = widgets.HTML(value = "Exact Match", layout = widgets.Layout(margin = "0px 0px -8px 0px")).add_class("heading5").add_class("medium")
        self.mustInclude = SearchBar(header = "Must include all", header2="(AND)", placeholder='e.g. “civil null” means each post in the result must contain the word “civil” and “null”')
        self.containOneOf = SearchBar(header = "Must include one of", header2="(OR)", placeholder='e.g. “census penny” means each post in the result must contain either “census” or “penny” or both')
        self.exclude = SearchBar(header = "Must not include", header2="(NOT)", placeholder='e.g. “toxic ban” means none of the posts in the result contains the word “toxic” and “ban”')
        self.semanticSearch = SemanticSearch(placeholder = "e.g. misinformation and miscommunication")
        self.searches = widgets.VBox([widgets.HTML(value = "<b>Search Criteria</b>"), self.semanticSearch, self.keyWordSearch, self.mustInclude, self.containOneOf, self.exclude])
        self.searches.add_class("search-box")
        self.closeButton = widgets.Button(description = 'X')
        self.closeButton.add_class("close-button")
        self.advancedPage = widgets.VBox([self.closeButton, self.searches, self.bottomBar]).add_class("advanced-page")

class RandomSampleTab():
    def toggleSimilarityScore(self, change=None):
        self.tweetDisplay.displayAddOn = self.displaySimilarityScore.value
        if self.displaySimilarityScore.hidden > 0:
            self.tweetDisplay.displayAddOn = 0

    def __init__(self):
        self.tweetDisplay = TweetDisplay(height = "60vh", displayAddOn = 0, addOnColumnName = "SimilarityScore")
        self.sortBar = SortBar()
        self.displaySimilarityScore = ToggleSwitch(label = "Relevance", hidden=1, value=0).add_class("tweet-display-add-on")
        self.displaySimilarityScore.observe(self.toggleSimilarityScore, ["value"])

        # optionsBar = widgets.Box(children = [self.sortBar])
        # optionsBar.layout = widgets.Layout(align_items = "center", justify_content = "space-between", width = "100%")
        # self.searchedKeywords = ParameterDisplay(firstWord = "Searched", secondWord = "Keywords", headers = ["Must Include", "Contain one of", "Exclude"], notFound = 'To enter keywords, click "Search & Filter"')
        # self.appliedFilters = ParameterDisplay(firstWord = "Applied", secondWord = "Filters", headers = ["calendar.svg", "geography.svg", "username.svg", "repost.svg", "weight.svg"], notFound = 'To enter filters, click "Search & Filter"')

        self.sampleTitle = widgets.HTML().add_class("display-count")
        self.sampleSelector = SampleSelector(label="Generate New Sample >")
        self.sampleTopBar = widgets.HBox([self.sampleTitle, self.sampleSelector], layout=widgets.Layout(justify_content="space-between", flex="0 0"))
        self.sortingBar = widgets.HBox([self.sortBar, self.displaySimilarityScore], layout=widgets.Layout(flex="0 0"))
        self.randomSelection = widgets.VBox([self.sampleTopBar, self.sortingBar, self.tweetDisplay], layout=widgets.Layout(max_height="100%"))

    
class AISummaryModule():
    def __init__(self):
        self.loadingPage = LoadingPage(text="Generating AI Summary")
        self.aiSummary = AiSummary()
        self.title = widgets.HTML().add_class("display-count")
        self.pageSelect = PageSelect()
        self.summaryDisplay = TweetDisplay(height="60vh", noResultsVersion=1)
        self.leftBar = widgets.VBox([widgets.HTML("AI Generated Summary*").add_class("heading4").add_class("medium"), self.aiSummary, self.pageSelect]).add_class("left-bar")
        self.newSummaryButton = widgets.Button(description="Generate Another Summary").add_class("generic-button").add_class("summary-button")
        self.displayTitle = widgets.HTML("Contributing Posts").add_class("heading4").add_class("medium")
        self.summaryContent = widgets.HBox([self.leftBar, widgets.VBox([self.displayTitle, self.summaryDisplay]).add_class("right-bar")])
        self.summaryContent.add_class("summary-tab")
        self.summaryTab = widgets.VBox([self.title, self.summaryContent, self.newSummaryButton], layout=widgets.Layout(height="100%"))


class StanceAnalysisModule():
    def __init__(self):
        self.stanceAnalysis = StanceAnalysis()
        self.lastSearchedStances = []
        self.modifyStanceButton = widgets.Button(description="< Modify Stance Annotation", layout=widgets.Layout(margin="0 auto 0 0", flex="0 0")).add_class("clear-button")
        
        self.sampleSelector = SampleSelector(label="New Stance Analysis >", options=[50, 100, 200, 500, 1000])
        self.loadingScreen = LoadingPage(text="Applying Stance Annotation")
        self.title = widgets.HTML().add_class("display-count")
        self.stance0CheckBox = widgets.Checkbox(value=True, indent=False).add_class("stance-checkbox0")
        self.stance1CheckBox = widgets.Checkbox(value=True, indent=False).add_class("stance-checkbox1")
        self.stance2CheckBox = widgets.Checkbox(value=True, indent=False).add_class("stance-checkbox2")
        self.stance3CheckBox = widgets.Checkbox(value=True, indent=False).add_class("stance-checkbox3")
        self.defaultStanceCheckBox = widgets.Checkbox(value=True, indent=False, description="System Default - Irrelevant").add_class("stance-checkbox-1")
        self.checkboxes = [self.stance0CheckBox, self.stance1CheckBox, self.stance2CheckBox, self.stance3CheckBox, self.defaultStanceCheckBox]
        
        self.checkboxBar = widgets.HBox(self.checkboxes, layout=widgets.Layout(flex="0 0"))
        self.topBar = widgets.HBox([self.title, self.sampleSelector], layout=widgets.Layout(flex="0 0", justify_content="space-between"))
        self.sortBar = SortBar(columns=["None", "Date", "Geography", "Retweets", "Username", "Stance"], columnNames=["None", "CreatedTime", "State", "Retweets", "SenderScreenName", "stance"])
        
        self.tweetDisplay = TweetDisplay(displayAddOn=0, colorCode=1)
        
        self.stanceCorrections = []
        self.stanceCorrectionNums = []
        self.cancelModification = widgets.Button(description="Cancel").add_class("clear-button")
        
        self.retrainButton = widgets.Button(description="Update Classification").add_class("generic-button")
        
        self.popUp = widgets.HBox([self.cancelModification, self.retrainButton]).add_class("pop-up-options").add_class("stance-pop-up")
        self.stanceAnalysisResults = widgets.VBox([self.modifyStanceButton, self.topBar, self.checkboxBar, self.sortBar, self.tweetDisplay]).add_class("stance-vbox")


class TimeSeriesModule():
    def __init__(self):
        self.timeSeriesMode = widgets.ToggleButtons(options = ["Overview", "Sex", "Age", "Education", "Income", "Party", "Ideology", "Urbanicity", "Metro", "Stance"]).add_class("time-series-toggle")
        
        # 现有的复选框
        self.genderCheckboxes = [widgets.Checkbox(value=True, description = "male"), widgets.Checkbox(value=True, description = "female"), ]
        self.ageCheckboxes = [
            widgets.Checkbox(value=True, description = "18-29"), 
            widgets.Checkbox(value=True, description = "30-44"), 
            widgets.Checkbox(value=True, description = "45-59"), 
            widgets.Checkbox(value=True, description = "60+")
        ]
        self.educationCheckboxes = [
            widgets.Checkbox(value=True, description = "Less than high school"), 
            widgets.Checkbox(value=True, description = "High school graduate (high school diploma or the equivalent GED)"), 
            widgets.Checkbox(value=True, description = "Some college or Associate's degree"), 
            widgets.Checkbox(value=True, description = "Bachelor's degree or higher")
        ]

        # 新增的复选框
        self.incomeCheckboxes = [
            widgets.Checkbox(value=True, description = "Low income"), 
            widgets.Checkbox(value=True, description = "Middle income"), 
            widgets.Checkbox(value=True, description = "High income")
        ]
        
        self.partyCheckboxes = [
            widgets.Checkbox(value=True, description = "Republican"), 
            widgets.Checkbox(value=True, description = "Democrat"), 
            widgets.Checkbox(value=True, description = "Independent")
        ]
        
        self.ideologyCheckboxes = [
            widgets.Checkbox(value=True, description = "Conservative"), 
            widgets.Checkbox(value=True, description = "Moderate"), 
            widgets.Checkbox(value=True, description = "Liberal")
        ]
        
        self.urbanicityCheckboxes = [
            widgets.Checkbox(value=True, description = "Urban"), 
            widgets.Checkbox(value=True, description = "Suburban"), 
            widgets.Checkbox(value=True, description = "Rural")
        ]
        
        self.metroCheckboxes = [
            widgets.Checkbox(value=True, description = "Metro"), 
            widgets.Checkbox(value=True, description = "Non-Metro")
        ]

        self.stanceCheckboxes = []

        self.graphTitle = widgets.HTML("Number of posts across time").add_class("heading4").add_class("medium")
        self.graphSubtitle = widgets.HTML("Total number of posts: ").add_class("body1")
        self.timeSeries = widgets.VBox([self.timeSeriesMode, self.graphTitle, self.graphSubtitle]).add_class("time-series")
        self.stanceNote = widgets.HTML("No stance information available.\nPlease use the 'Stance Analysis' tab to generate a stance analysis.")

class WordCloudModule():
    def __init__(self):
        self.title = widgets.HTML("")
        self.title.add_class("heading4").add_class("medium")
        self.searchBar = SearchBar(header = "<b>Exclude words</b>", placeholder = "")
        self.regenerate = widgets.Button(description="Regenerate word cloud").add_class("generic-button")
        self.tools = widgets.VBox([self.searchBar, self.regenerate]).add_class("word-cloud-tools")
        self.content = widgets.HBox([self.tools]).add_class("word-cloud-content")
        self.page = widgets.VBox([self.title, self.content]).add_class("word-cloud-page")

class InferDemographicsModule():
    def __init__(self):
        self.inferDemographics = InferDemographics()
        self.page = self.inferDemographics

        self.backButton = widgets.Button(description="< New Demographic Analysis", layout=widgets.Layout(margin="0 auto 0 0", flex="0 0")).add_class("clear-button")
        self.resultsTitle = widgets.HTML("").add_class("display-count")
        self.resultsHTML = widgets.HTML("")
        self.resultsPage = widgets.VBox([self.backButton, self.resultsTitle, self.resultsHTML])

    def showResults(self, dists, sampleSize):
        self.resultsTitle.value = f"Inferred demographics for {sampleSize:,d} posts"
        parts = []
        for attr, counts in dists.items():
            total = sum(counts.values()) or 1
            rows = "".join(
                f"<tr><td>{html.escape(str(value))}</td><td>{count}</td><td>{count / total:.0%}</td></tr>"
                for value, count in sorted(counts.items(), key=lambda kv: -kv[1])
            )
            parts.append(
                f"<h4>{html.escape(attr.replace('_', ' ').title())}</h4>"
                f"<table><tr><th>Value</th><th>Posts</th><th>Share</th></tr>{rows}</table>"
            )
        self.resultsHTML.value = "<div class='demographics-results'>" + "".join(parts) + "</div>"