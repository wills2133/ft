$(function(){
     $('#tabs a').click(function(e) {
          e.preventDefault();                
          $('#tabs li').removeClass("current").removeClass("hoverItem");
          $(this).parent().addClass("current");
          $("#content div").removeClass("show");
          $('#' + $(this).attr('title')).addClass('show');
      });

     $('#tabs a').hover(function(){
        if(!$(this).parent().hasClass("current")){
          $(this).parent().addClass("hoverItem");
        }
     },function(){
        $(this).parent().removeClass("hoverItem");
     });
  });