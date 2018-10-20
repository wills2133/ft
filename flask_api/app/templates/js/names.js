
var jsonFile = "./data/category.json";
var medicineFile = "./data/medicine.json";
var toxicsFile = "./data/toxics.json";

var selectionBig;

var prescription_total = []
var correct_usage = []
function hoverDrop(args){
    selectionBig = args.innerHTML;
}
var form = d3.select("#names").append("form");
var category_form = d3.select("#tags").append("form");
var num_clicks = 0            
function clickDrop(args) {
    var selection = args.innerHTML;
    
    d3.select("#names").selectAll("input").remove();
    d3.select("#names").selectAll("br").remove();
  //  d3.select("#tags").selectAll("input").remove();
  //  d3.select("#detail".text).remove();
    
    
    d3.json(jsonFile, function(error,data_group){
        //show medicine names
        
     //   console.log(data_group)
        form.html(" ")
            .selectAll("input")
            .data(data_group[selectionBig][selection])
            .enter()
            .append("input")
            .attr("type","button")
            .attr("class","name-btn")
            .attr("name", "otherName")
            .attr("value", function(d){return d})
            .on("click",function(d){
                 var coor = d3.mouse(this) 
                 var medicine_name = this.value;
                 var prescription = [];
                 
                 var composition = new Object();
                 var correct_usage1 = new Object()
                 composition.name = medicine_name
                 
                 num_clicks = num_clicks + 1
                 d3.select(".veri-button").style("visibility","visible")
               
                 d3.json(medicineFile, function(error,medicine_data){
                     var medicine_array = medicine_data["medicine"];
                     var category = Object.keys(medicine_array[10]);
                         category = category.slice(1,category.length);
                         category.unshift(medicine_name);
                         category = category.slice(0,5).concat(category.slice(7,category.length))
                    
                     medicine_array.forEach(function(d){
                        if (d["药名"] == medicine_name){
                            correct_usage1.name = medicine_name
                            correct_usage1.max = d["最大剂量"]
                            correct_usage1.min = d["最小剂量"]
                          //  details.push("药理")
                          //  details.push(d["药理"]) 
                            
                        }
                    })
                    correct_usage.push(correct_usage1)
                    
                    
                         
                    category_form.append("label")
                                 .attr("class","btn")
                                 .text(medicine_name)
                                 .on("click",function(d){
                                    d3.select("#details").selectAll("div").remove()
                                    d3.select("#details")
                                      .style("visibility","hidden")
                                    d3.select("#details")
                                       .attr("height",100)
                                       .attr("width",100)
                                       .style("background-color","#f0f0f0")
                                   //    .style("visibility","hidden");
                                    
                                    
                                    d3.select("#details")
                                      .transition()
                                      .style("visibility","visible")
                                      .attr("width",300)
                                      .attr("height",400)
                                      .style("overflow-y","scroll")
                                      
                                      
                                      var details = [] //saves details of medicines
                    console.log(medicine_name,"medicine name")
                    medicine_array.forEach(function(d){
                        if (d["药名"] == medicine_name){
                            details.push(medicine_name)
                            details.push("异名")
                            details.push(d["异名"])
                            details.push("药性")
                            details.push(d["药性"])
                            details.push("功效主治")
                            details.push(d["功效主治"])
                            details.push("用法用量")
                            details.push(d["用法用量"])
                            details.push("使用注意")
                            details.push(d["使用注意"])
                            details.push("附方")
                            details.push(d["附方"])
                            
                          //  details.push(d["药理"]) 
                            
                        }
                    })
                                  //  var instruction_form = d3.select("#instruction")
                                     d3.select("#details").selectAll("div")
                                    .data(details).enter()
                                    .append("div")
                                    
                                       .text(function(d){return d})
                                       .style("color","black")
                                       .style("color",function(d,i){if(i%2 == 1){return "red"}})
                                       .style("background-color",function(d,i){if(i==0){return "#b1e0c6"}})
                                       .style("font-size","12px")
                                  //     .style("text-decoration",function(d,i){if (i == 0){return "underline"}})
                                       .style("text-decoration",function(d,i){if (i%2 == 1){return "underline"}})
                                       
                                      // .style("font-size",function(d,i){if(i==0){return "15px"}})
                                    //   .style("font-family", function(d,i){if(i==0){return "Verdana"}})
                                                             
                                       
                                      
                                      
                  // d3.select("#details").style("overflow","auto")
                                   })
                    category_form.append("input")
                                 .attr("class","input-box")
                                 .attr("type","text")
                                 .on("focusout",function(d){
                                    composition.dose = this.value;
                                    prescription_total.push(composition)
                                    console.log("composition",composition,"********")
                                    })
                                
                 //   document.getElementsByClassName("input-box").node().focus()
                    category_form.append("label")
                                 .text("g")
                    if (num_clicks==4) {
                        category_form = d3.select("#tags").append("form");
                        num_clicks = 0
                    }
                    
                    
                    
                                   
        
        
        })  //end of medicine.json file
                 
                 
              })  //end of click function
    
    })
    
    
    
} //end of function clickDrop
    
function prescription() {
    console.log(prescription_total)
    console.log(correct_usage)
    var medicine_feedback = []
    var toxics_feedback = []
    prescription_total.forEach(function(d,i){
        var tmp = new Object()
       // console.log(Number(d.dose),Number(correct_usage[i].max),Number(d.dose) > Number(correct_usage[i].max))
        if (Number(d.dose) > Number(correct_usage[i].max)) {
            tmp.name = d.name
            tmp.feedback = "根据药典，超过常用计量的最大用量值"
            medicine_feedback.push(tmp)
        }
        else if(Number(d.dose) == Number(correct_usage[i].max)) {
            tmp.name = d.name
            tmp.feedback = "根据药典，达到常用计量的最大用量值"
            medicine_feedback.push(tmp)
        }
        
    })
    
   
    
    d3.json(toxicsFile,function(error, toxics_data){
         big = toxics_data["大毒"];
         medium = toxics_data["有毒"]
         small = toxics_data["小毒"]
        
    
        
        prescription_total.forEach(function(d,i){
            tmp = new Object()
            big.forEach(function(t,j){
                if (d.name == t) {
                    tmp.name = d.name
                    tmp.feedback = "有大毒。"
                     toxics_feedback.push(tmp)
                }
            })
            medium.forEach(function(tt,jj){
                if (d.name == tt) {
                    tmp.name = d.name
                    tmp.feedback = "有毒。"
                     toxics_feedback.push(tmp)
                }
            })
            small.forEach(function(ttt,jjj){
                if (d.name == ttt) {
                    tmp.name = d.name
                    tmp.feedback = "有小毒。"
                     toxics_feedback.push(tmp)
                }
            })
        })
    
    console.log(toxics_feedback) 
    d3.select("#feedback").selectAll("div")
      .data(toxics_feedback)
      .enter()
      .append("div")
      .text(function(d){console.log(d,"********");return d.name+"："+"  "+d.feedback})
      .style("color","red")
        
    })    
        
    console.log(toxics_feedback) 
    d3.select("#feedback").selectAll("div")
      .data(medicine_feedback)
      .enter()
      .append("div")
      .text(function(d){console.log(d,"********");return d.name+"："+"  "+d.feedback})
      .style("color","red")
   
    
}
