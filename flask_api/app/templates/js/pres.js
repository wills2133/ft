
d3.select("#presButton").on("click",function(d){
    var jsonFile = "./data/casesTest.json";
    var button_text = ["查看医案","查看医案","查看医案","查看医案"]
    d3.json(jsonFile,function(error,data){
       
        var caseArray = data.cases
        d3.select("#pres").selectAll("div")
                          .data(caseArray).enter()
                          .append("div")
                          .style("position","absolute")
                          .style("top",function(d,i){return 50*i+600 + "px"})
                          .style("width",900)
                          .style("height",100)
                          .style("border-radius","5px")
                          .style("margin-right","5px")
                                    
                                       .text(function(d){return d["处方"]})
                                       .style("display","inline-block")
                                       .style("color","black")
                                     //  .style("background-color","#d9b38c")
                                       .style("font-size","18px");
                                  //     .style("text-decoration",function(d,i){if (i == 0){return "underline"}})
        d3.select("#button").selectAll("div")
                                    .data(button_text).enter()
                                    .append("div")
                                    .attr("class","pres-button")
                                    .style("position","absolute")
                                    .style("top",function(d,i){return 50*i+600 + "px"})
                                    .style("width",900)
                                    .style("height",100)
                                    .style("border-radius","5px")
                                    .style("margin-right","5px")
                                    .text(function(d){return d})
                                    .style("display","inline-block")
                                     //  .style("background-color","#d9b38c")
                                    .style("font-size","18px")
                                    .on("click",function(d,i){
                                        d3.select("#case")
                                        
                                        .text("姓名："+caseArray[i]["姓名"]+"； "+
                                              "性别："+caseArray[i]["性别"]+"; "+
                                              "年龄："+caseArray[i]["年龄"]+"; "+
                                              "主诉及病史："+caseArray[i]["主诉及病史"]+"; "+
                                              "诊查："+caseArray[i]["诊查"]+"; "+
                                              "辨证："+caseArray[i]["辨证"]+"; "+
                                              "治法："+caseArray[i]["治法"]+"; "+
                                              "处方："+caseArray[i]["处方"]+"; "
                                              )
                                        .style("backgound-color","#d9b38c")
                                        .style("color","#A52A2A")
                                        })
                    
           })
});
