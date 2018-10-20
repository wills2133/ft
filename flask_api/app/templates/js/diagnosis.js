 $(document).ready(function() { 
    $(".sel").select2({ width: '17%' });           
});

d3.select("#diagnosis").on("click", function(){
   //  console.log($('.sel').select2('data'));
 //    document.getElementsByClassName('sel');

function symptoms() {
   var symptomArray = [];
  //  var sel = document.getElementsByClassName('sel');
   var sel = $('.sel option:selected').text();
   symptomArray = sel.split(" ");
 //  console.log(symptomArray)
    
    
    return symptomArray;
}

function compareTwoArrays(a,b) {
   same = []
   for(var i = 0; i < a.length; i++){
      for (var j = 0; j < b.length; j++){
         if (a[i] == b[j]) {
            same.push(a[i])
         }
      }
   }
   return same
}
   radio_ids_gender = ["nan","nv"]
   radio_ids_age = ["below14","14to45","45to70","above70"]
   var gender = "不明"
   for (var i = 0; i<radio_ids_gender.length; i++){
     // console.log(radio_ids_gender[i])
      var element = document.getElementById(radio_ids_gender[i])
      if (element.checked) {
       gender = element.value
      }
   }
   var age = "不明"
   for (var i = 0; i<radio_ids_age.length; i++){
      var element = document.getElementById(radio_ids_age[i])
      if (element.checked) {
       age = element.value
      }
   }
   
   
    var symptomArray = symptoms();
    symptomArray = symptomArray.filter(function(str) {
    return /\S/.test(str);
});
    console.log("symptomArray",symptomArray)
    
    d3.json("./data/synonym.json",function (error,data_synonym){
      var pulse = ["脉浮","脉散","脉芤","脉革","脉牢","脉动","脉沉","脉伏","脉迟","脉数","脉洪",
                   "脉细","脉实","脉虚","脉微","脉缓","脉弦","脉紧","脉滑","脉涩","脉濡","脉促",
                   "脉结","脉代","脉弱","脉短","脉长","脉疾"];
      var keys = Object.keys(data_synonym).concat(pulse) 
      var new_patient_vec = []
      if (symptomArray.length == 0) {
         window.alert("请选择症状！")
      }

    
   //   console.log(keys)
     // console.log(keys.indexOf("头痛"))
      for (var i = 0; i < symptomArray.length; i++) {
        
        new_patient_vec.push(keys.indexOf(symptomArray[i]));
    }
    
    console.log(new_patient_vec,"new patient")
    
      var jsonFile = "./data/neiKe.json";
      d3.json(jsonFile,function(error, data){
        var disease_array = data["内科"];
        var main_vec = [];
        var disease = [];
        var zheng = []
        disease_array.forEach(function(d){
           
           main_vec.push(d["主证向量"]);
           disease.push(d["病名"]);
           zheng.push(d["证症"])
        }) //end of forEach
      console.log(zheng[0][0]["证名"],"22222")
      
      // diagnoize main disease name
      var main_same = []
      for (var i = 0; i< main_vec.length; i++){
         var main_same_tmp = []
         main_same_tmp = compareTwoArrays(new_patient_vec, main_vec[i])         
         if (main_same_tmp.length != 0) {
            main_same.push({  //save the index and the vector of shared symptoms between new patient and diagnosis standard.
               "idx":i,
               "symptoms":main_same_tmp,
            })
       //     console.log(main_same_tmp,"********")
         }
      }
      
      var disease_name = ""
      var max_length_idx
      var same_length = []
      var similarity_main = []
      console.log("main_same",main_same)
      if (main_same.length != 0) {
         console.log("right")
        for (var i = 0; i < main_same.length; i++){
            same_length.push(main_same[i]["symptoms"].length);
            var same_length_tmp = main_same[i]["symptoms"].length.toFixed(2);
            var main_vec_selected = main_vec[main_same[i]["idx"]].length.toFixed(2);
            var similarity_tmp = (same_length_tmp/new_patient_vec.length.toFixed(2) + same_length_tmp/main_vec_selected)/2.0;
       //     console.log(similarity_tmp)
            similarity_main.push(similarity_tmp)
         //   var diagnosis_disease = disease[main_same[i]["idx"]]
      
    //  console.log(diagnosis_disease,"*******")

         }
      var max_similarity_disease = Math.max.apply(null,similarity_main)
      
      max_length_idx = similarity_main.indexOf(max_similarity_disease)
      console.log(max_length_idx,similarity_main)
   //   if (main_same) {
        //code
    //  }
      var diagnosis_disease_idx = main_same[max_length_idx]["idx"]
      var diagnosis_disease = disease[diagnosis_disease_idx]
      
      console.log(diagnosis_disease)
      }
      else{
         diagnosis_disease = ""
         var max_similarity_disease = -1
      }
         
      
   // diagnose zheng
   var main_same_zheng = []
   for (var i=0; i < zheng.length; i++){ 
    //  console.log(zheng[i],disease[i],i)
      for (var j=0; j < 1; j++) {
         var zheng_vec_total = zheng[i][j]["体症向量"]
         for (var k = 0; k < zheng_vec_total.length; k++){ //k is 
            for(var t = 0; t < zheng_vec_total[k].length; t++){
               var main_same_tmp = []
               main_same_tmp = compareTwoArrays(new_patient_vec, zheng_vec_total[k][t])
             //  console.log(zheng_vec_total[k][t],"888888888")
               if (main_same_tmp.length != 0) {
                   main_same_zheng.push({  //save the index and the vector of shared symptoms between new patient and diagnosis standard.
                   "idx_i":i, //i is the index for disease
                   "idx_j":j, //j is the index for stage
                   "idx_k":k, //k is the index for zheng
                   "idx_t":t, //t is the index for different combination of symptoms
                   "symptoms":main_same_tmp,
               })
            }
            
            }
         }
      }
   }
   console.log("main_same2",main_same_zheng)
   
      
      
      var max_length_zheng = 0
      var same_length = []
      var similarity_zheng = []
      if (main_same_zheng.length == 0 && diagnosis_disease == "") {
         window.alert("此症候群还在学习中... ...")
      }
      else{
         for (var x = 0; x < main_same_zheng.length; x++){ // as long as one can find index x of main_same, he can get all the i, j, k, t indices
            same_length.push(main_same_zheng[x]["symptoms"].length);
            var same_length_tmp = main_same_zheng[x]["symptoms"].length.toFixed(2); // the length of the shared vector between new patient and the diagnosis standard
            var idx_i = main_same_zheng[x]["idx_i"]
            var idx_j = main_same_zheng[x]["idx_j"]
            var idx_k = main_same_zheng[x]["idx_k"]
            var idx_t = main_same_zheng[x]["idx_t"]
            
            var zheng_vec_selected = zheng[idx_i][idx_j]["体症向量"][idx_k][idx_t].length.toFixed(2)
            
            var similarity_zheng_tmp = (same_length_tmp/new_patient_vec.length.toFixed(2) + same_length_tmp/zheng_vec_selected)/2.0;
//console.log(same_length_tmp,new_patient_vec.length.toFixed(2),zheng_vec_selected,same_length_tmp/new_patient_vec.length.toFixed(2), same_length_tmp/zheng_vec_selected,similarity_zheng_tmp,"note")
            similarity_zheng.push(similarity_zheng_tmp)


         }
      }
      
      var max_similarity_zheng = Math.max.apply(null,similarity_zheng)
      
      max_length_zheng = similarity_zheng.indexOf(max_similarity_zheng)
      //we calculate similarity based on symptoms in zheng and find the most similar one, now we will calculate similarity of main symptom vector
      //based on the index obtained from zheng calculations.
      var disease_vec_selected = main_vec[main_same_zheng[max_length_zheng]["idx_i"]].length.toFixed(2);
    //  console.log(disease_vec_selected,main_same_zheng[max_length_zheng]["idx_i"])
      if (disease_vec_selected > 0) {
        var max_similarity_disease2 = (same_length_tmp/new_patient_vec.length.toFixed(2) + same_length_tmp/disease_vec_selected)/2.0;
      }
      else{
         max_similarity_disease2 = 0
      }
      
      console.log(max_similarity_disease,max_similarity_zheng,max_similarity_disease2)
      
      // compare the max similarity values calculated from indices generated from zheng and main symptoms.
      var threshold_main_symptom = 0.6
      var similarity_zheng2 = []
      var main_same_zheng_subset = []
      if ((max_similarity_disease - max_similarity_disease2)>threshold_main_symptom) {
         //calculate the similarity of zheng for the diagnosed disease.
         for (var x = 0; x < main_same_zheng.length; x++) {
            
            if (main_same_zheng[x]["idx_i"] == diagnosis_disease_idx) {
               console.log(disease[main_same_zheng[x]["idx_i"] ])
                var same_length_tmp = main_same_zheng[x]["symptoms"].length.toFixed(2)
                idx_i = main_same_zheng[x]["idx_i"]
                idx_j = main_same_zheng[x]["idx_j"]
                idx_k = main_same_zheng[x]["idx_k"]
                main_same_zheng_subset.push({  //save the index and the vector of shared symptoms between new patient and diagnosis standard.
                   "idx_i":diagnosis_disease_idx, //i is the index for disease
                   "idx_j":idx_j, //j is the index for stage
                   "idx_k":idx_k, //k is the index for zheng
                   "symptoms":main_same_zheng[x]["symptoms"],
               })
            
                
               // console.log(zheng[idx_i][idx_j]["证名"])
                var zheng_vec_selected2 = zheng[idx_i][idx_j]["体症向量"][idx_k].length.toFixed(2)
                var similarity_zheng_tmp = (same_length_tmp/new_patient_vec.length.toFixed(2) + same_length_tmp/zheng_vec_selected)/2.0
                similarity_zheng2.push(similarity_zheng_tmp)//make sure the indices of main_same_zheng(_subset) are consistent with similaity_zheng(2)
            };
         };
        
        var max_similarity_zheng2 = Math.max.apply(null,similarity_zheng2)
      //  console.log(similarity_zheng2,"similarity_zheng2")
        var max_length_zheng2 = similarity_zheng2.indexOf(max_similarity_zheng2)
      //  console.log("&&&&&&&&&",max_length_zheng2,max_similarity_zheng2,similarity_zheng2,"&&&&&&&&&")
        var diagnosis_zheng = zheng[diagnosis_disease_idx]
                                 [main_same_zheng_subset[max_length_zheng2]["idx_j"]]["证名"]
                                 [main_same_zheng_subset[max_length_zheng2]["idx_k"]]
        
      }
      else{
         var diagnosis_zheng = zheng[main_same_zheng[max_length_zheng]["idx_i"]]
                                 [main_same_zheng[max_length_zheng]["idx_j"]]["证名"]
                                 [main_same_zheng[max_length_zheng]["idx_k"]]
         var diagnosis_disease = disease[[main_same_zheng[max_length_zheng]["idx_i"]]]
      };
      
      //get prescription
      var prescription = ""
      d3.json("./data/yanfang.json",function (error,data_prescription){
         var data_prescription_array = data_prescription["内科"]
         for (var i = 0; i < data_prescription_array.length; i++){
            if (data_prescription_array[i]["病名"] == diagnosis_disease &&
                data_prescription_array[i]["证名"] == diagnosis_zheng) {
                prescription = data_prescription_array[i]["药方"]
            }
         }
         
      if (prescription == "") {
         prescription = "该药方正在学习中... ..."
      }
      
       var svg = d3.select('#result');
       var font_size = 20
       var font_color = "#696969"
       svg.selectAll("text").remove()
       svg.append("rect")
          .attr("width","100%")
          .attr("height",380)
          .attr("class","bounding-rect")
          .attr("rx", 60)
          .attr("ry", 60)
          
      svg.append('text')
         .attr('x', 65)
         .attr('y', 65)
         .attr("class","text")
         .text("患者信息:")
         .attr('font-size', 22)
         .attr("fill","#8B4513")
         
      svg.append('text')
         .attr('x', 65)
         .attr('y', 95)
         .text("性别："+" "+gender)
         .attr('font-size', font_size)
         .attr("fill",font_color)
      
      svg.append('text')
         .attr('x', 65)
         .attr('y', 125)
         .text("年龄："+" "+ age)
         .attr('font-size', font_size)
         .attr("fill",font_color)
      svg.append('text')
         .attr('x', 65)
         .attr('y', 155)
         .text("症状："+" "+symptomArray)
         .attr('font-size', font_size)
         .attr("fill",font_color)    
      svg.append('text')
         .attr('x', 65)
         .attr('y', 195)
         .attr("class","text")
         .text("诊断结果:")
         .attr('font-size', 22)
         .attr("fill","#8B4513")
      svg.append('text')
         .attr('x', 65)
         .attr('y', 225)
         .text("病名："+" "+diagnosis_disease)
         .attr('font-size', font_size)
         .attr("fill",font_color)
         
      svg.append('text')
         .attr('x', 65)
         .attr('y', 255)
         .text("证型："+" "+ diagnosis_zheng)
         .attr('font-size', font_size)
         .attr("fill",font_color)
                    
      svg.append('text')
         .attr('x', 65)
         .attr('y', 295)
         .attr("class","text")
         .text("推荐验方:")
         .attr('font-size', 22)
         .attr("fill","#8B4513")
         
      svg.append('text')
         .attr('x', 65)
         .attr('y', 325)
         .text("验方："+" "+ prescription)
         .attr('font-size', font_size)
         .attr("fill",font_color)
        
      d3.select("#prescription")
         .append("div")
         .attr("class","cover-button cover-button2")
         .text("验方加减")
         .on("click",function(d){
            window.open("medicine.html")
            

         })
      
 
       })//end of json yanfang.json
      
        
      }) //end of json neiKe.json
    
    
    
    }) //end of synonym.json
    
    

    
      

    
    
})//end of click