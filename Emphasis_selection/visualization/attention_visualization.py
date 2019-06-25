# Credits to Lin Zhouhan(@hantek) for the visualization code
# The code is customized for this task
import random, os, numpy, scipy
from codecs import open


# if (k%2 == 0) {
# var heat_text = "<p><br><b>_</b><br>";
# } else {
# var heat_text = "<b>Example:</b><br>";
# }
def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    fileName = "visualization/" + fileName
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
     var color = "255, 70, 50";

    for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {

            intensity[i] = trigram_weights[k][i];

            if (intensity[i] > max_intensity) {
                max_intensity = intensity[i];
            }
            if (intensity[i] < min_intensity) {
                min_intensity = intensity[i];
            }
        }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }

    var heat_text = "<p><b>_ </b>";
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n" % (",".join(map(str, weights)))
    # print("weightsString:", weightsString)
    fOut.write(part1)
    # print("textsString: ", textsString)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()

    return
